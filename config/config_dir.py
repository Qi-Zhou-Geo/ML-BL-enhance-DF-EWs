#!/usr/bin/python
# -*- coding: UTF-8 -*-

#__modification time__ = 2024-09-22
#__author__ = Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
#__find me__ = qi.zhou@gfz-potsdam.de, qi.zhou.geo@gmail.com, https://github.com/Nedasd
# Please do not distribute this code without the author's permission

import os

def config_dir():
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # get the parent path
    output_dir = f"{parent_dir}/output"
    sac_path_Illgraben = f"/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/EU/Illgraben/"
    sac_path_Museum = f"/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/USA/Museum_Fire/"
    sac_path_Luding = "/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/Luding"

    CONFIG_dir = {
        "parent_dir": parent_dir,
        "output_dir": output_dir,
        "sac_path_Illgraben": sac_path_Illgraben,
        "sac_path_Museum": sac_path_Museum,
        "sac_path_Luding": sac_path_Luding
    }

    return CONFIG_dir

# please keep in mind this file I/O directory
CONFIG_dir = config_dir()
