#!/usr/bin/python
# -*- coding: UTF-8 -*-txt')
# written by Qi Zhou, Helmholtz Centre Potsdam - GFZ German Research Centre for Geosciences
import pandas as pd
import numpy as np
import argparse

def load_data(data_path, model_type, feature_type, station_list):
    
    ILL18Pro = pd.read_csv(f"{data_path}{station_list[0]}{model_type}_{feature_type}_testOut.txt", header=0)
    date = np.array(ILL18Pro.iloc[:, 0])
    ILL18Pro = np.array(ILL18Pro.iloc[:, -1])

    ILL12Pro = pd.read_csv(f"{data_path}{station_list[1]}{model_type}_{feature_type}_testOut.txt", header=0)
    ILL12Pro = np.array(ILL12Pro.iloc[:, -1])

    ILL13Pro = pd.read_csv(f"{data_path}{station_list[0]}{model_type}_{feature_type}_testOut.txt", header=0)
    ILL13Pro = np.array(ILL13Pro.iloc[:, -1])
    
    assert date.size == ILL18Pro.size, f" date.size != ILL18Pro"
    assert ILL18Pro.size == ILL12Pro.size, f" date.size != ILL18Pro"
    assert ILL12Pro.size == ILL13Pro.size, f" date.size != ILL18Pro"

    return date, ILL18Pro, ILL12Pro, ILL13Pro


def write_down_warning(model_type, feature_type):

    attention_len1 = 5 # for (ILL18, ILL12) and (ILL12, ILL13) 
    attention_len2 = 10 # for ILL12
    pro_threshold = 0.8

    station_list = ["ILL18", "ILL12", "ILL13"]
    date, ILL18Pro, ILL12Pro, ILL13Pro = load_data(DATA_DIR, model_type, feature_type, station_list)


    for step in range(attention_len2, ILL18Pro.size):
        # go through all the data
        proILL18 = ILL18Pro[(step- attention_len1) : step]
        proILL12 = ILL12Pro[(step- attention_len1) : step]
        proILL13 = ILL13Pro[(step- attention_len1) : step]

        proILL12_longer = ILL12Pro[(step- attention_len2) : step]

        count_ILL18 = np.sum(proILL18 > pro_threshold)
        count_ILL12 = np.sum(proILL12 > pro_threshold)
        count_ILL13 = np.sum(proILL13 > pro_threshold)

        count_ILL12_longer = np.sum(proILL12_longer > pro_threshold)

        status = "noise"
        if count_ILL18 >=5 and count_ILL12 >=5:
            status = "warning"
        else:
            pass

        if count_ILL12 >= 5 and count_ILL13 >= 5:
            status = "warning"
        else:
            pass

        if count_ILL12_longer >= 10:
            status = "warning"
        else:
            pass

        record = f"{step}, {date[step]}, " \
                 f"{ILL18Pro[step]:.4f}, {count_ILL18}, " \
                 f"{ILL12Pro[step]:.4f}, {count_ILL12}, " \
                 f"{ILL13Pro[step]:.4f}, {count_ILL13}," \
                 f"{count_ILL12_longer}, {status}"

        f = open(f"{OUTPUT_DIR}/{model_type}_{feature_type}_final_networt_warning_all.txt", 'a')
        f.write(str(record) + "\n")
        f.close()

        if status == "warning":
            print(record)

            f = open(f"{OUTPUT_DIR}/{model_type}_{feature_type}_final_networt_warning_event.txt", 'a')
            f.write(str(record) + "\n")
            f.close()



def main(model, feature):
    global DATA_DIR, OUTPUT_DIR
    DATA_DIR = "/home/qizhou/3paper/2AGU/"
    OUTPUT_DIR = "/home/qizhou/3paper/2AGU/3modelMigration/out15/"
    write_down_warning(model, feature)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='please input the following param')

    parser.add_argument("--model", default="RF", type=str, help="model type")
    parser.add_argument("--feature", default="rf", type=str, help="feature type")

    args = parser.parse_args()

    main(args.model, args.feature)


date, ILL18Pro, ILL12Pro, ILL13Pro = load_data(data_path = "/Users/qizhou/Desktop/RF_C/", 
                                              model_type = "RF", 
                                              feature_type = "C", 
                                              station_list = ["ILL18", "ILL12", "ILL13"])
