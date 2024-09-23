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

    CONFIG_dir = {
        "parent_dir": parent_dir,
        "output_dir": output_dir
    }

    return CONFIG_dir

CONFIG_dir = config_dir()
