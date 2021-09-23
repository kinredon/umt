from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__D = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg_d = __D
#
# Training options
# with regard to pascal, the directories under the path will be ./VOC2007, ./VOC2012"
__D.PASCAL = "path_to/dataset/VOCdevkit"
__D.PASCAL_CYCLECLIPART = (
    "path_to/dataset/voc2clip/voc_cycle_clipart"
)
__D.PASCAL_CYCLE_WATER = "path_to/dataset/voc2water/voc_cycle_water"
__D.PASCALCLIP = ""
__D.PASCALWATER = "path_to/dataset/VOCdevkit"


# For these datasets, the directories under the path will be Annotations  ImageSets  JPEGImages."
__D.KITTI = "path_to/dataset/city2kitti/kitti_voc/VOC2007"
__D.KITTI_CYCLE_CITY = (
    "path_to/dataset/city2kitti/kitti_cycle_city_10/VOC2007"
)
__D.CITYSCAPE_CAR_CYCLE_KITTI = (
    "path_to/dataset/city2kitti/city_cycle_kitti_10"
)
__D.CLIPART = "path_to/dataset/voc2clip/clipart"
__D.CLIPART_CYCLEVOC = (
    "path_to/dataset/voc2clip/clipart_cycle_voc/VOC2007"
)
__D.WATER = "path_to/dataset/watercolor2k"
__D.WATER_CYCLE_VOC = "path_to/dataset/voc2water/water_cycle_voc"
__D.SIM10K = "path_to/dataset/sim10k/VOC2007"
__D.SIM10K_CYCLE_CITY = "path_to/dataset/sim10k2city/sim10k_cycle_city_voc_10"
__D.CITYSCAPE_CAR = "path_to/dataset/city2foggy/city_voc"
__D.CITYSCAPE_CAR_CYCLE_SIM10K = (
    "path_to/dataset/sim10k2city/city_cycle_sim10k_voc_10"
)
__D.CITYSCAPE = "path_to/dataset/city2foggy/city_voc"
__D.CITYSCAPE_CYCLE_FOGGY = (
    "path_to/dataset/city2foggy/city_cycle_foggy_voc"
)
__D.FOGGYCITY = "path_to/dataset/city2foggy/foggy_voc"
__D.FOGGYCITY_CYCLE_CITY = (
    "path_to/dataset/city2foggy/foggy_cycle_city_voc"
)


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError("{} is not a valid config key".format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(
                    ("Type mismatch ({} vs. {}) " "for config key: {}").format(
                        type(b[k]), type(v), k
                    )
                )

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(("Error under config key: {}".format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml

    with open(filename, "r") as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __D)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval

    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split(".")
        d = __D
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(
            d[subkey]
        ), "type {} does not match original type {}".format(
            type(value), type(d[subkey])
        )
        d[subkey] = value
