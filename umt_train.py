# coding:utf-8
# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pprint
import pdb
import time
import _init_paths

import torch
from torch.autograd import Variable
import torch.nn as nn


from model.utils.config import cfg, cfg_from_file, cfg_from_list
from model.utils.net_utils import (
    adjust_learning_rate,
    save_checkpoint,
    get_dataloader,
    setup_seed,
)
from model.ema.optim_weight_ema import WeightEMA
from model.utils.parser_func import parse_args, set_dataset_args
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv

from prettytimer import PrettyTimer


def get_cfg():
    args = parse_args()

    print("Called with args:")
    print(args)
    args = set_dataset_args(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print("Using config:")
    pprint.pprint(cfg)
    # np.random.seed(cfg.RNG_SEED)
    setup_seed(cfg.RNG_SEED)
    return args


if __name__ == "__main__":
    args = get_cfg()

    output_dir = f"{args.save_dir}/{args.net}/{args.dataset}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset_t == "water":
        args.aug = False

    if args.dataset_t == "foggy_cityscape":
        # initilize the network here.
        from model.umt_faster_rcnn_truncate.umt_vgg16 import vgg16
        from model.umt_faster_rcnn_truncate.umt_resnet import resnet
    else:
        from model.umt_faster_rcnn.umt_vgg16 import vgg16
        from model.umt_faster_rcnn.umt_resnet import resnet

    student_save_name = os.path.join(
        output_dir,
        "conf_{}_conf_gamma_{}_source_like_{}_aug_{}_target_like_{}_pe_{}_pl_{}_thresh_{}"
        "_lambda_{}_student_target_{}".format(
            args.conf,
            args.conf_gamma,
            args.source_like,
            args.aug,
            args.target_like,
            args.pretrained_epoch,
            args.pl,
            args.threshold,
            args.lam,
            args.dataset_t,
        ),
    )
    print("Model will be saved to: ")
    print(student_save_name)
    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    # source train set
    s_imdb, s_train_size, s_dataloader = get_dataloader(args.imdb_name, args)

    # source-like/fake-source train set data loader
    if args.source_like:
        s_fake_imdb, s_fake_train_size, s_fake_dataloader = get_dataloader(
            args.imdb_name_fake_source, args, sequential=True, augment=args.aug
        )
    else:
        s_fake_imdb, s_fake_train_size, s_fake_dataloader = get_dataloader(
            args.imdb_name_target, args, sequential=True, augment=args.aug
        )
    # target train set
    t_imdb, t_train_size, t_dataloader = get_dataloader(
        args.imdb_name_target, args, sequential=True, augment=args.aug
    )
    # target-like/fake-target train set
    t_fake_imdb, t_fake_train_size, t_fake_dataloader = get_dataloader(
        args.imdb_name_fake_target, args
    )

    print("{:d} source roidb entries".format(s_train_size))
    print("{:d} source like roidb entries".format(s_fake_train_size))
    print("{:d} target roidb entries".format(t_train_size))
    print("{:d} target like roidb entries".format(t_fake_train_size))

    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)
    # ship to cuda
    if args.cuda:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    im_data = Variable(im_data)
    im_info = Variable(im_info)
    num_boxes = Variable(num_boxes)
    gt_boxes = Variable(gt_boxes)

    if args.cuda:
        cfg.CUDA = True

    imdb = s_imdb

    if args.net == "vgg16":
        student_fasterRCNN = vgg16(
            imdb.classes,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
        teacher_fasterRCNN = vgg16(
            imdb.classes,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
    elif args.net == "res101":
        student_fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
        teacher_fasterRCNN = resnet(
            imdb.classes,
            101,
            pretrained=True,
            class_agnostic=args.class_agnostic,
            conf=args.conf,
        )
    elif args.net == "res50":
        student_fasterRCNN = resnet(
            imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
        teacher_fasterRCNN = resnet(
            imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic
        )
    else:
        print("network is not defined")
        pdb.set_trace()

    student_fasterRCNN.create_architecture()
    teacher_fasterRCNN.create_architecture()

    lr = cfg.TRAIN.LEARNING_RATE
    lr = args.lr

    student_detection_params = []
    params = []
    for key, value in dict(student_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            if "bias" in key:
                params += [
                    {
                        "params": [value],
                        "lr": lr * (cfg.TRAIN.DOUBLE_BIAS + 1),
                        "weight_decay": cfg.TRAIN.BIAS_DECAY
                        and cfg.TRAIN.WEIGHT_DECAY
                        or 0,
                    }
                ]
            else:
                params += [
                    {
                        "params": [value],
                        "lr": lr,
                        "weight_decay": cfg.TRAIN.WEIGHT_DECAY,
                    }
                ]
            student_detection_params += [value]

    teacher_detection_params = []
    for key, value in dict(teacher_fasterRCNN.named_parameters()).items():
        if value.requires_grad:
            teacher_detection_params += [value]
            value.requires_grad = False

    if args.optimizer == "adam":
        lr = lr * 0.1
        student_optimizer = torch.optim.Adam(params)
    elif args.optimizer == "sgd":
        student_optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
    teacher_optimizer = WeightEMA(
        teacher_detection_params, student_detection_params, alpha=args.teacher_alpha
    )

    if args.cuda:
        student_fasterRCNN.cuda()
        teacher_fasterRCNN.cuda()

    if args.resume:
        student_checkpoint = torch.load(args.student_load_name)
        args.session = student_checkpoint["session"]
        args.start_epoch = student_checkpoint["epoch"]
        student_fasterRCNN.load_state_dict(student_checkpoint["model"])
        student_optimizer.load_state_dict(student_checkpoint["optimizer"])
        lr = student_optimizer.param_groups[0]["lr"]
        if "pooling_mode" in student_checkpoint.keys():
            cfg.POOLING_MODE = student_checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (args.student_load_name))

        teacher_checkpoint = torch.load(args.teacher_load_name)
        teacher_fasterRCNN.load_state_dict(teacher_checkpoint["model"])
        if "pooling_mode" in teacher_checkpoint.keys():
            cfg.POOLING_MODE = teacher_checkpoint["pooling_mode"]
        print("loaded checkpoint %s" % (args.teacher_load_name))

    if args.mGPUs:
        student_fasterRCNN = nn.DataParallel(student_fasterRCNN)
        teacher_fasterRCNN = nn.DataParallel(teacher_fasterRCNN)
    iters_per_epoch = int(10000 / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter

        logger = SummaryWriter("logs")

    count_iter = 0
    conf_gamma = args.conf_gamma
    pretrained_epoch = args.pretrained_epoch
    timer = PrettyTimer()
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        # setting to train mode
        student_fasterRCNN.train()
        teacher_fasterRCNN.train()
        loss_temp = 0

        start = time.time()
        epoch_start = time.time()
        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(student_optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        data_iter_s = iter(s_dataloader)
        data_iter_t = iter(t_dataloader)
        data_iter_s_fake = iter(s_fake_dataloader)
        data_iter_t_fake = iter(t_fake_dataloader)
        for step in range(1, iters_per_epoch + 1):
            timer.start("iter")
            try:
                data_s = next(data_iter_s)
            except:
                data_iter_s = iter(s_dataloader)
                data_s = next(data_iter_s)

            try:
                data_s_fake = next(data_iter_s_fake)
            except:
                data_iter_s_fake = iter(s_fake_dataloader)
                data_s_fake = next(data_iter_s_fake)

            try:
                data_t = next(data_iter_t)
            except:
                data_iter_t = iter(t_dataloader)
                data_t = next(data_iter_t)

            assert (
                data_s_fake[0].size() == data_t[0].size()
            ), "The size should be same between source fake and target"
            assert (
                data_s_fake[1] == data_t[1]
            ).all(), "The image info should be same between source fake and target"
            try:
                data_t_fake = next(data_iter_t_fake)
            except:
                data_iter_t_fake = iter(t_fake_dataloader)
                data_t_fake = next(data_iter_t_fake)

            # eta = 1.0
            count_iter += 1

            # put source data into variable
            im_data.data.resize_(data_s[0].size()).copy_(data_s[0])
            im_info.data.resize_(data_s[1].size()).copy_(data_s[1])
            gt_boxes.data.resize_(data_s[2].size()).copy_(data_s[2])
            num_boxes.data.resize_(data_s[3].size()).copy_(data_s[3])

            student_fasterRCNN.zero_grad()
            (
                rois,
                cls_prob,
                bbox_pred,
                rpn_loss_cls,
                rpn_loss_box,
                RCNN_loss_cls,
                RCNN_loss_bbox,
                rois_label,
                out_d_pixel,
                out_d,
                confidence_loss,
                _,
            ) = student_fasterRCNN(im_data, im_info, gt_boxes, num_boxes, hints=True)
            loss = (
                rpn_loss_cls.mean()
                + rpn_loss_box.mean()
                + RCNN_loss_cls.mean()
                + RCNN_loss_bbox.mean()
            )
            if args.conf:
                conf_loss = confidence_loss.mean()

            if args.target_like:
                # put fake target data into variable
                im_data.data.resize_(data_t_fake[0].size()).copy_(data_t_fake[0])
                im_info.data.resize_(data_t_fake[1].size()).copy_(data_t_fake[1])
                # gt is empty
                gt_boxes.data.resize_(data_t_fake[2].size()).copy_(data_t_fake[2])
                num_boxes.data.resize_(data_t_fake[3].size()).copy_(data_t_fake[3])

                (
                    rois,
                    cls_prob,
                    bbox_pred,
                    rpn_loss_cls_t_fake,
                    rpn_loss_box_t_fake,
                    RCNN_loss_cls_t_fake,
                    RCNN_loss_bbox_t_fake,
                    rois_label_t_fake,
                    out_d_pixel,
                    out_d,
                    _,
                    _,
                ) = student_fasterRCNN(
                    im_data, im_info, gt_boxes, num_boxes, hints=False
                )  # --------------------------------
                loss += (
                    rpn_loss_cls_t_fake.mean()
                    + rpn_loss_box_t_fake.mean()
                    + RCNN_loss_cls_t_fake.mean()
                    + RCNN_loss_bbox_t_fake.mean()
                )

            if epoch > pretrained_epoch and args.pl:
                teacher_fasterRCNN.eval()

                im_data.data.resize_(data_s_fake[0].size()).copy_(data_s_fake[0])
                im_info.data.resize_(data_s_fake[1].size()).copy_(data_s_fake[1])
                # gt is emqpty
                gt_boxes.data.resize_(1, 1, 5).zero_()
                num_boxes.data.resize_(1).zero_()
                (
                    rois,
                    cls_prob,
                    bbox_pred,
                    rpn_loss_cls_,
                    rpn_loss_box_,
                    RCNN_loss_cls_,
                    RCNN_loss_bbox_,
                    rois_label_,
                    d_pred_,
                    _,
                    _,
                    confidence_s_fake,
                ) = teacher_fasterRCNN(im_data, im_info, gt_boxes, num_boxes, test=True)

                scores = cls_prob.data
                boxes = rois.data[:, :, 1:5]

                if cfg.TEST.BBOX_REG:
                    # Apply bounding-box regression deltas
                    box_deltas = bbox_pred.data
                    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                        # Optionally normalize targets by a precomputed mean and stdev
                        if args.class_agnostic:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                            box_deltas = box_deltas.view(1, -1, 4)
                        else:
                            box_deltas = (
                                box_deltas.view(-1, 4)
                                * torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_STDS
                                ).cuda()
                                + torch.FloatTensor(
                                    cfg.TRAIN.BBOX_NORMALIZE_MEANS
                                ).cuda()
                            )
                            box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                    pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                    pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
                else:
                    # Simply repeat the boxes, once for each class
                    pred_boxes = np.tile(boxes, (1, scores.shape[1]))

                scores = scores.squeeze()
                if args.conf:
                    scores = torch.sqrt(
                        scores * confidence_s_fake
                    )  # using confidence score to adjust scores
                pred_boxes = pred_boxes.squeeze()
                gt_boxes_target = []
                pre_thresh = 0.0
                thresh = args.threshold
                empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))
                for j in range(1, len(imdb.classes)):
                    inds = torch.nonzero(scores[:, j] > pre_thresh).view(-1)
                    # if there is det
                    if inds.numel() > 0:
                        cls_scores = scores[:, j][inds]
                        _, order = torch.sort(cls_scores, 0, True)
                        if args.class_agnostic:
                            cls_boxes = pred_boxes[inds, :]
                        else:
                            cls_boxes = pred_boxes[inds][:, j * 4 : (j + 1) * 4]

                        cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                        # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                        cls_dets = cls_dets[order]
                        keep = nms(cls_dets, cfg.TEST.NMS)
                        cls_dets = cls_dets[keep.view(-1).long()]
                        # all_boxes[j][i] = cls_dets.cpu().numpy()
                        cls_dets_numpy = cls_dets.cpu().numpy()
                        for i in range(np.minimum(10, cls_dets_numpy.shape[0])):
                            bbox = tuple(
                                int(np.round(x)) for x in cls_dets_numpy[i, :4]
                            )
                            score = cls_dets_numpy[i, -1]
                            if score > thresh:
                                gt_boxes_target.append(list(bbox[0:4]) + [j])

                gt_boxes_padding = torch.FloatTensor(cfg.MAX_NUM_GT_BOXES, 5).zero_()
                if len(gt_boxes_target) != 0:
                    gt_boxes_numpy = torch.FloatTensor(gt_boxes_target)
                    num_boxes_cpu = torch.LongTensor(
                        [min(gt_boxes_numpy.size(0), cfg.MAX_NUM_GT_BOXES)]
                    )
                    gt_boxes_padding[:num_boxes_cpu, :] = gt_boxes_numpy[:num_boxes_cpu]
                else:
                    num_boxes_cpu = torch.LongTensor([0])

                # teacher_fasterRCNN.train()
                # put source-like data into variable
                im_data.data.resize_(data_t[0].size()).copy_(data_t[0])
                im_info.data.resize_(data_t[1].size()).copy_(data_t[1])
                gt_boxes_padding = torch.unsqueeze(gt_boxes_padding, 0)
                gt_boxes.data.resize_(gt_boxes_padding.size()).copy_(gt_boxes_padding)
                num_boxes.data.resize_(num_boxes_cpu.size()).copy_(num_boxes_cpu)

                (
                    rois,
                    cls_prob,
                    bbox_pred,
                    rpn_loss_cls_s_fake,
                    rpn_loss_box_s_fake,
                    RCNN_loss_cls_s_fake,
                    RCNN_loss_bbox_s_fake,
                    rois_label_s_fake,
                    out_d_pixel,
                    out_d,
                    _,
                    _,
                ) = student_fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                loss += args.lam * (
                    rpn_loss_cls_s_fake.mean()
                    + rpn_loss_box_s_fake.mean()
                    + RCNN_loss_cls_s_fake.mean()
                    + RCNN_loss_bbox_s_fake.mean()
                )

            if args.conf:
                loss += conf_gamma * conf_loss

            loss_temp += loss.item()
            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
            teacher_fasterRCNN.zero_grad()
            teacher_optimizer.step()
            timer.end("iter")
            estimate_time = timer.eta(
                "iter", count_iter, args.max_epochs * iters_per_epoch
            )
            if step % args.disp_interval == 0:
                end = time.time()
                if step > 0:
                    loss_temp /= args.disp_interval

                if args.mGPUs:
                    loss_rpn_cls = rpn_loss_cls.mean().item()
                    loss_rpn_box = rpn_loss_box.mean().item()
                    loss_rcnn_cls = RCNN_loss_cls.mean().item()
                    loss_rcnn_box = RCNN_loss_bbox.mean().item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt
                    if args.pl and epoch > pretrained_epoch:
                        loss_rpn_cls_s_fake = rpn_loss_cls_s_fake.mean().item()
                        loss_rpn_box_s_fake = rpn_loss_box_s_fake.mean().item()
                        loss_rcnn_cls_s_fake = RCNN_loss_cls_s_fake.mean().item()
                        loss_rcnn_box_s_fake = RCNN_loss_bbox_s_fake.mean().item()
                        fg_cnt_s_fake = torch.sum(rois_label_s_fake.data.ne(0))
                        bg_cnt_s_fake = rois_label_s_fake.data.numel() - fg_cnt_s_fake
                    if args.target_like:
                        loss_rpn_cls_t_fake = rpn_loss_cls_t_fake.mean().item()
                        loss_rpn_box_t_fake = rpn_loss_box_t_fake.mean().item()
                        loss_rcnn_cls_t_fake = RCNN_loss_cls_t_fake.mean().item()
                        loss_rcnn_box_t_fake = RCNN_loss_bbox_t_fake.mean().item()
                        fg_cnt_t_fake = torch.sum(rois_label_t_fake.data.ne(0))
                        bg_cnt_t_fake = rois_label_t_fake.data.numel() - fg_cnt_t_fake

                    # dloss_s_fake = dloss_s_fake.mean().item()
                    # dloss_t_fake = dloss_t_fake.mean().item()
                    # dloss_s_p_fake = dloss_s_p_fake.mean().item()
                    # dloss_t_p_fake = dloss_t_p_fake.mean().item()
                else:
                    loss_rpn_cls = rpn_loss_cls.item()
                    loss_rpn_box = rpn_loss_box.item()
                    loss_rcnn_cls = RCNN_loss_cls.item()
                    loss_rcnn_box = RCNN_loss_bbox.item()
                    fg_cnt = torch.sum(rois_label.data.ne(0))
                    bg_cnt = rois_label.data.numel() - fg_cnt

                    if args.conf:
                        loss_conf = conf_loss.item()

                    if args.pl and epoch > pretrained_epoch:
                        loss_rpn_cls_s_fake = rpn_loss_cls_s_fake.item()
                        loss_rpn_box_s_fake = rpn_loss_box_s_fake.item()
                        loss_rcnn_cls_s_fake = RCNN_loss_cls_s_fake.item()
                        loss_rcnn_box_s_fake = RCNN_loss_bbox_s_fake.item()
                        fg_cnt_s_fake = torch.sum(rois_label_s_fake.data.ne(0))
                        bg_cnt_s_fake = rois_label_s_fake.data.numel() - fg_cnt

                    if args.target_like:
                        loss_rpn_cls_t_fake = rpn_loss_cls_t_fake.item()
                        loss_rpn_box_t_fake = rpn_loss_box_t_fake.item()
                        loss_rcnn_cls_t_fake = RCNN_loss_cls_t_fake.item()
                        loss_rcnn_box_t_fake = RCNN_loss_bbox_t_fake.item()
                        fg_cnt_t_fake = torch.sum(rois_label_t_fake.data.ne(0))
                        bg_cnt_t_fake = rois_label_t_fake.data.numel() - fg_cnt_t_fake

                print(
                    "[session %d][epoch %2d][iter %4d/%4d] lr: %.2e, loss: %.4f, eta: %s"
                    % (
                        args.session,
                        epoch,
                        step,
                        iters_per_epoch,
                        lr,
                        loss_temp,
                        estimate_time,
                    )
                )
                print(
                    "\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end - start)
                )
                print(
                    "\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f"
                    % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box)
                )
                if args.pl and epoch > pretrained_epoch:
                    print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt_s_fake, bg_cnt_s_fake))
                    print(
                        "\t\t\trpn_cls_s_fake: %.4f, rpn_box_s_fake: %.4f, rcnn_cls_s_fake: %.4f, rcnn_box_s_fake %.4f"
                        % (
                            loss_rpn_cls_s_fake,
                            loss_rpn_box_s_fake,
                            loss_rcnn_cls_s_fake,
                            loss_rcnn_box_s_fake,
                        )
                    )

                if args.target_like:
                    print("\t\t\tfg/bg=(%d/%d)" % (fg_cnt_t_fake, bg_cnt_t_fake))
                    print(
                        "\t\t\trpn_cls_t_fake: %.4f, rpn_box_t_fake: %.4f, rcnn_cls_t_fake: %.4f, rcnn_box_t_fake %.4f"
                        % (
                            loss_rpn_cls_t_fake,
                            loss_rpn_box_t_fake,
                            loss_rcnn_cls_t_fake,
                            loss_rcnn_box_t_fake,
                        )
                    )
                if args.conf is True:
                    print(f"\t\t\tconf loss: {loss_conf:.4}")

                if args.use_tfboard:
                    info = {
                        "loss": loss_temp,
                        "loss_rpn_cls": loss_rpn_cls,
                        "loss_rpn_box": loss_rpn_box,
                        "loss_rcnn_cls": loss_rcnn_cls,
                        "loss_rcnn_box": loss_rcnn_box,
                        "loss_rpn_cls_s_fake": loss_rpn_cls_s_fake,
                        "loss_rpn_box_s_fake": loss_rpn_box_s_fake,
                        "loss_rcnn_cls_s_fake": loss_rcnn_cls_s_fake,
                        "loss_rcnn_box_s_fake": loss_rcnn_box_s_fake,
                        "loss_rpn_cls_t_fake": loss_rpn_cls_t_fake
                        if args.target_like is True
                        else 0,
                        "loss_rpn_box_t_fake": loss_rpn_box_t_fake
                        if args.target_like is True
                        else 0,
                        "loss_rcnn_cls_t_fake": loss_rcnn_cls_t_fake
                        if args.target_like is True
                        else 0,
                        "loss_rcnn_box_t_fake": loss_rcnn_box_t_fake
                        if args.target_like is True
                        else 0,
                        "loss_conf": loss_conf if args.conf is True else 0,
                        "conf_gamma": conf_gamma,
                    }
                    logger.add_scalars(
                        "logs_s_{}/losses".format(args.session),
                        info,
                        (epoch - 1) * iters_per_epoch + step,
                    )

                loss_temp = 0

                start = time.time()

        student_save_name = os.path.join(
            output_dir,
            "conf_{}_conf_gamma_{}_source_like_{}_aug_{}_target_like_{}_pe_{}_pl_{}_"
            "thresh_{}_lambda_{}_lam2_{}_student_target_{}_session_{}_epoch_{}_step_{}.pth".format(
                args.conf,
                args.conf_gamma,
                args.source_like,
                args.aug,
                args.target_like,
                args.pretrained_epoch,
                args.pl,
                args.threshold,
                args.lam,
                args.lam2,
                args.dataset_t,
                args.session,
                epoch,
                step,
            ),
        )
        save_checkpoint(
            {
                "session": args.session,
                "epoch": epoch + 1,
                "model": student_fasterRCNN.mumt_train.pyodule.state_dict()
                if args.mGPUs
                else student_fasterRCNN.state_dict(),
                "optimizer": student_optimizer.state_dict(),
                "pooling_mode": cfg.POOLING_MODE,
                "class_agnostic": args.class_agnostic,
            },
            student_save_name,
        )
        print("save student model: {}".format(student_save_name))

        teacher_save_name = os.path.join(
            output_dir,
            "conf_{}_conf_gamma_{}_source_like_{}_aug_{}_target_like_{}_pe_{}_pl_{}_"
            "thresh_{}_lambda_{}_lam2_{}_teacher_target_{}_session_{}_epoch_{}_step_{}.pth".format(
                args.conf,
                args.conf_gamma,
                args.source_like,
                args.aug,
                args.target_like,
                args.pretrained_epoch,
                args.pl,
                args.threshold,
                args.lam,
                args.lam2,
                args.dataset_t,
                args.session,
                epoch,
                step,
            ),
        )
        save_checkpoint(
            {
                "session": args.session,
                "epoch": epoch + 1,
                "model": teacher_fasterRCNN.mumt_train.pyodule.state_dict()
                if args.mGPUs
                else teacher_fasterRCNN.state_dict(),
                "pooling_mode": cfg.POOLING_MODE,
                "class_agnostic": args.class_agnostic,
            },
            teacher_save_name,
        )
        print("save teacher model: {}".format(teacher_save_name))
        epoch_end = time.time()
        print("epoch cost time: {} min".format((epoch_end - epoch_start) / 60.0))

        # cmd = (
        #     f"python test_net_global_local.py --dataset {args.dataset_t} --net {args.net}"
        #     f" --load_name {student_save_name}"
        # )
        # print("cmd: ", cmd)
        # cmd = [i.strip() for i in cmd.split(" ") if len(i.strip()) > 0]
        # try:
        #     proc = subprocess.Popen(cmd)
        #     proc.wait()
        # except (KeyboardInterrupt, SystemExit):
        #     pass

        # cmd = (
        #     f"python test_net_global_local.py --dataset {args.dataset_t} --net {args.net}"
        #     f" --load_name {teacher_save_name}"
        # )
        # print("cmd: ", cmd)
        # cmd = [i.strip() for i in cmd.split(" ") if len(i.strip()) > 0]
        # try:
        #     proc = subprocess.Popen(cmd)
        #     proc.wait()
        # except (KeyboardInterrupt, SystemExit):
        #     pass

    if args.use_tfboard:
        logger.close()
