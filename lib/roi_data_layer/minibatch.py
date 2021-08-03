# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Xinlei Chen
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
from scipy.misc import imread
from scipy.misc import imsave
from model.utils.config import cfg
from model.utils.blob import prep_im_for_blob, im_list_to_blob
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from torchvision import transforms as T


def get_minibatch(roidb, num_classes, seg_return=False, augment=False, seed=2020):
    """Given a roidb, construct a minibatch sampled from it."""
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    random_scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                                    size=num_images)
    assert (cfg.TRAIN.BATCH_SIZE % num_images == 0), \
        'num_images ({}) must divide BATCH_SIZE ({})'. \
            format(num_images, cfg.TRAIN.BATCH_SIZE)

    # Get the input image blob, formatted for caffe
    im_blob, im_scales, gt_boxes = _get_image_blob(roidb, random_scale_inds, augment=augment, seed=seed)

    assert len(im_scales) == 1, "Single batch only"

    blobs = {'data': im_blob}
    blobs['gt_boxes'] = gt_boxes
    blobs['im_info'] = np.array(
        [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
        dtype=np.float32)
    if seg_return:
        blobs['seg_map'] = roidb[0]['seg_map']
    blobs['img_id'] = roidb[0]['img_id']
    blobs['path'] = roidb[0]['image']

    return blobs

def bbs2numpy(bbs):
    bboxes = []
    for bb in bbs.bounding_boxes:
        x1 = bb.x1 - 1
        y1 = bb.y1 - 1
        w = bb.x2 - bb.x1
        h = bb.y2 - bb.y2
        label = float(bb.label)
        bboxes.append([x1, y1, w, h, label])
    return np.array(bboxes, dtype=np.float32)

def _get_image_blob(roidb, scale_inds, augment=False, seed=2020):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    assert len(roidb) == 1, "Single batch only"

    # gt boxes: (x1, y1, x2, y2, cls)
    if cfg.TRAIN.USE_ALL_GT:
        # Include all ground truth boxes
        gt_inds = np.where(roidb[0]['gt_classes'] != 0)[0]
    else:
        # For the COCO ground truth boxes, exclude the ones that are ''iscrowd''
        gt_inds = np.where((roidb[0]['gt_classes'] != 0) & np.all(roidb[0]['gt_overlaps'].toarray() > -1.0, axis=1))[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :]
    # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
    gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]

    num_images = len(roidb)

    processed_ims = []
    im_scales = []
    for i in range(num_images):
        # im = cv2.imread(roidb[i]['image'])
        im = imread(roidb[i]['image'])
        # print(roidb[i]['image'])
        if len(im.shape) == 2:
            im = im[:, :, np.newaxis]
            im = np.concatenate((im, im, im), axis=2)
        # flip the channel, since the original one using cv2
        # rgb -> bgr
        im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        # data augmentation
        if augment:
            im, gt_boxes = augmentor(im, gt_boxes, seed=seed)
        # imsave("target_aug.jpg", im[:, :, ::-1])
        target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, target_size,
                                        cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale)
        processed_ims.append(im)
        gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale


    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_scales, gt_boxes

def augmentor(image, bounding_boxes, seed=2020):

    ia.seed(seed)
    bbxes = []
    for gt_box in bounding_boxes:
        x1 = gt_box[0] + 1
        y1 = gt_box[1] + 1
        x2 = gt_box[2] + x1
        y2 = gt_box[3] + y1
        bbxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=str(gt_box[4])))
    bbs = BoundingBoxesOnImage(bbxes, shape=image.shape)
    seq = iaa.Sequential([
        iaa.CropAndPad(
            percent=(-0.05, 0.1),
            pad_mode=ia.ALL,
            pad_cval=(0, 255)
        )
    ])
    image, bbs_aug = seq(image=image, bounding_boxes=bbs)
    jitter_param = 0.4
    transform = T.Compose([
        T.ToPILImage(),
        T.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
    ])
    image_pil = transform(image)
    image = np.array(image_pil)
    bbs_aug = bbs_aug.remove_out_of_image().clip_out_of_image()
    gt_boxes = bbs2numpy(bbs_aug)
    return image, gt_boxes