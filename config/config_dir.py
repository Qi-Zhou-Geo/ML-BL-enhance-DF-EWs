#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-09-22
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os
import platform
import sys

# define the parent directory
if platform.system() == 'Darwin':
    parent_dir = "/Users/qizhou/#python/#GitHub_saved/ML-BL-enhance-DF-EWs"
elif platform.system() == 'Linux':
    parent_dir = "/home/qizhou/3paper/2AGU_revise/ML-BL-enhance-DF-EWs"
else:
    print(f"check the parent_dir for platform.system() == {platform.system()}")
# add the parent_dir to the sys
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
else:
    pass


def config_dir(parent_dir):
    sac_dir = f"/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic"
    output_dir = f"{parent_dir}/output"
    feature_output_dir = f"/storage/vast-gfz-hpc-01/home/qizhou/3paper/0seismic_feature"

    CONFIG_dir = {
        "sac_dir": sac_dir,
        "parent_dir": parent_dir,
        "output_dir": output_dir,
        "feature_output_dir":feature_output_dir,
    }

    return CONFIG_dir


def path_mapping(seismic_network):

    mapping = {"9J": "EU/Illgraben", # EU data
               "9S": "EU/Illgraben",
               # USA data
               "CI":"USA/Montecito",
               "1A": "USA/Museum_Fire",
               "Y7": "USA/Chalk_Cliffs",
               # Chinese data
               "DC": "AS/Dongchuan",
               "MJ": "AS/MinJiang",
               "CC": "AS/Goulinping",
               "XF": "AS/Ramche",
               "XN": "AS/Bothekoshi",
               # New Zealand data
               "MR": "OC/Ruapehu",
               }

    dir = mapping.get(seismic_network, "check function path_mapping")

    return dir



# please keep in mind this file I/O directory
CONFIG_dir = config_dir(parent_dir)
