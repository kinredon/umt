# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.pascal_voc_water import pascal_voc_water
from datasets.pascal_voc_cyclewater import pascal_voc_cyclewater
from datasets.pascal_voc_cycleclipart import pascal_voc_cycleclipart
from datasets.sim10k import sim10k
from datasets.water import water
from datasets.clipart import clipart
from datasets.clipart_cyclevoc import clipart_cyclevoc
from datasets.sim10k_cycle import sim10k_cycle
from datasets.cityscape import cityscape
from datasets.cityscape_car import cityscape_car
from datasets.foggy_cityscape import foggy_cityscape
from datasets.city_cycle_foggy import city_cycle_foggy
from datasets.foggy_cycle_city import foggy_cycle_city
from datasets.sim10k_cycle_city import sim10k_cycle_city
from datasets.cityscape_car_cycle_sim10k import cityscape_car_cycle_sim10k
from datasets.pascal_voc_cycle_water import pascal_voc_cycle_water
from datasets.water_cycle_voc import water_cycle_voc
from datasets.pascal_voc_7cls import pascal_voc_7cls

# cityscape to kitti
from datasets.kitti import kitti
from datasets.kitti_cycle_city import kitti_cycle_city
from datasets.cityscape_car_cycle_kitti import cityscape_car_cycle_kitti

# voc 2 comic

from datasets.pascal_voc_7cls_comic import pascal_voc_7cls_comic
from datasets.comic import comic
from datasets.pascal_voc_cycle_comic import pascal_voc_cycle_comic
from datasets.comic_cycle_voc import comic_cycle_voc

# unit
from datasets.city_unit_foggy import city_unit_foggy
from datasets.foggy_unit_city import foggy_unit_city

for split in ["train", "trainval", "val", "test", "train_s"]:
    name = "cityscape_{}".format(split)
    __sets[name] = lambda split=split: cityscape(split)
for split in ["train", "trainval", "val", "test", "train_s"]:
    name = "city_cycle_foggy_{}".format(split)
    __sets[name] = lambda split=split: city_cycle_foggy(split)

for split in ["train", "trainval", "val", "test", "train_s"]:
    name = "city_unit_foggy_{}".format(split)
    __sets[name] = lambda split=split: city_unit_foggy(split)    

splites = ["train", "trainval", "val", "test"]
splites.extend(["test{}".format(str(i)) for i in range(11)])
for split in splites:
    name = "cityscape_car_{}".format(split)
    __sets[name] = lambda split=split: cityscape_car(split)


for split in ["train", "trainval", "val", "test"]:
    name = "cityscape_car_cycle_sim10k_{}".format(split)
    __sets[name] = lambda split=split: cityscape_car_cycle_sim10k(split)
for split in ["train", "trainval", "val", "test"]:
    name = "cityscape_car_cycle_kitti_{}".format(split)
    __sets[name] = lambda split=split: cityscape_car_cycle_kitti(split)

for split in ["train", "trainval", "val", "test"]:
    name = "kitti_{}".format(split)
    __sets[name] = lambda split=split: kitti(split)
for split in ["train", "trainval", "val", "test"]:
    name = "kitti_cycle_city_{}".format(split)
    __sets[name] = lambda split=split: kitti_cycle_city(split)

foggy_cityscape_splites = ["train", "trainval", "val", "test", "train_t", "test_t"]
foggy_cityscape_splites.extend(["test{}".format(str(i)) for i in range(11)])
for split in foggy_cityscape_splites:
    name = "foggy_cityscape_{}".format(split)
    __sets[name] = lambda split=split: foggy_cityscape(split)
for split in ["train", "trainval", "test", "train_t"]:
    name = "foggy_cycle_city_{}".format(split)
    __sets[name] = lambda split=split: foggy_cycle_city(split)

for split in ["train", "trainval", "test", "train_t"]:
    name = "foggy_unit_city_{}".format(split)
    __sets[name] = lambda split=split: foggy_unit_city(split)    

for split in ["train", "val"]:
    name = "sim10k_{}".format(split)
    __sets[name] = lambda split=split: sim10k(split)
for split in ["train", "val"]:
    name = "sim10k_cycle_city_{}".format(split)
    __sets[name] = lambda split=split: sim10k_cycle_city(split)

for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc(split, year)

for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_7cls_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_7cls(split, year)

    for year in ["2007", "2012"]:
        for split in ["train", "val", "trainval", "test"]:
            name = "voc_7cls_comic_{}_{}".format(year, split)
            __sets[name] = lambda split=split, year=year: pascal_voc_7cls_comic(
                split, year
            )

for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_water(split, year)
for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_cycleclipart_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_cycleclipart(
            split, year
        )
for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_cycle_water_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_cycle_water(
            split, year
        )
for year in ["2007", "2012"]:
    for split in ["train", "val", "trainval", "test"]:
        name = "voc_cycle_comic_{}_{}".format(year, split)
        __sets[name] = lambda split=split, year=year: pascal_voc_cycle_comic(
            split, year
        )

clipart_cityscape_splites = ["train", "trainval", "val", "test", "all"]
clipart_cityscape_splites.extend(["test{}".format(str(i)) for i in range(11)])
for year in ["2007"]:
    for split in clipart_cityscape_splites:
        name = "clipart_{}".format(split)
        __sets[name] = lambda split=split: clipart(split, year)
for year in ["2007"]:
    for split in ["trainval", "test"]:
        name = "clipart_cyclevoc_{}".format(split)
        __sets[name] = lambda split=split: clipart_cyclevoc(split, year)

for year in ["2007"]:
    for split in ["train", "test"]:
        name = "water_{}".format(split)
        __sets[name] = lambda split=split: water(split, year)
for year in ["2007"]:
    for split in ["train", "test"]:
        name = "water_cycle_voc_{}".format(split)
        __sets[name] = lambda split=split: water_cycle_voc(split, year)

for year in ["2007"]:
    for split in ["train", "test"]:
        name = "comic_{}".format(split)
        __sets[name] = lambda split=split: comic(split, year)

for year in ["2007"]:
    for split in ["train", "test"]:
        name = "comic_cycle_voc_{}".format(split)
        __sets[name] = lambda split=split: comic_cycle_voc(split, year)


def get_imdb(name):
    """Get an imdb (image database) by name."""
    if name not in __sets:
        raise KeyError("Unknown dataset: {}".format(name))
    return __sets[name]()


def list_imdbs():
    """List all registered imdbs."""
    return list(__sets.keys())
