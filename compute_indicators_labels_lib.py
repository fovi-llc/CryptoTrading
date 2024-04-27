from operator import concat
import os
import pandas as pd
from technical_analysis_lib import TecnicalAnalysis
import datetime
import random
from config import RUN as run_conf
from multiprocessing import pool


# https://stats.stackexchange.com/questions/312780/why-is-accuracy-not-the-best-measure-for-assessing-classification-models


def preprocess_filename(params):
    filename, RUN = params
    print(filename)
    if filename.split('.')[0] in RUN['off_label_set']:
        print("SKIPPING %s" % filename)
        return

    data = pd.read_csv(f"{RUN['folder']}{filename}")
    data = TecnicalAnalysis.compute_oscillators(data)
    data = TecnicalAnalysis.find_patterns(data)
    data = TecnicalAnalysis.add_timely_data(data)

    labels = pd.DataFrame()
    for bw in range(1, RUN['b_lim_sup_window']):
        for fw in range(1, RUN['f_lim_sup_window']):
            labels["lab_%d_%d" % (bw, fw)] = TecnicalAnalysis.assign_labels(data, bw, fw, RUN['alpha'], RUN['beta'])

    return data, labels



def preprocess(RUN):
    """
    Parallel preprocessing and labeling of coin datasets
    Save final dataset to a file in preprocessed_data folder
    :param RUN: configuration dict
    :return: 
    """
    jobs = pool.Pool(24)

    print("Preprocessing with: %s" % RUN)
    filenames = os.listdir(RUN['folder'])
    args = zip(filenames, [RUN] * len(filenames))
    args = [(k, v) for k, v in args]
    data_labels = jobs.map(preprocess_filename, args)
    jobs.terminate()

    data_list = [d[0] for d in data_labels]
    labels_list = [d[1] for d in data_labels]

    concat_data = pd.concat(data_list, ignore_index=True)
    concat_data['Date'] = pd.to_datetime(concat_data['Date'])
    
    concat_labels = pd.concat(labels_list, ignore_index=True)
    
    final_ds = pd.concat([concat_data, concat_labels], axis=1)
    final_ds = final_ds.dropna()
    final_ds.to_csv("processed_data/%strain_test_data.csv" % RUN['folder'].replace('/', '_'), index=False)



def get_dataset(RUN):
    """
    returns a dataset labeled with given forward and backward window
    :param RUN: run configuration dictionary
    :return: pandas dataframe wit 'label' column
    """
    
    ds = pd.read_csv("processed_data/%strain_test_data.csv" % RUN['folder'].replace('/', '_'))
    
    # remove off label data
    for coin in RUN['off_label_set']:
        ds = ds[ds['Asset_name'] != coin]
    
    fw = RUN['f_window']
    bw = RUN['b_window']
    label_col = "lab_%d_%d" % (bw, fw)
    
    labels = ds[label_col].copy()

    droped_lab = []
    for bw in range(1, RUN['b_lim_sup_window']):
        for fw in range(1, RUN['f_lim_sup_window']):
            label_col = "lab_%d_%d" % (bw, fw)
            droped_lab.append(label_col)
    
    ds = ds.drop(columns=droped_lab)
    
    ds['label'] = labels
    
    return ds

