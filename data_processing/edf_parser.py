# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 21:12:26 2019

@author: yijinli
"""
import pyedflib
import numpy as np
from mne import Epochs, pick_types, find_events
from mne.io import concatenate_raws, read_raw_edf
from datetime import datetime
import math
import ntpath
import os
from glob import glob

# reference: https://github.com/akaraspt/deepsleepnet/
channel = "EEG Fpz-Cz"
TIME_WINDOW_SIZE = 100
raw_data_folder = "input/"
output_folder = "output/"
if not os.path.exists(raw_data_folder):
    os.makedirs(raw_data_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

UNKNOWN = 5
annotationlabel = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
    "Sleep stage ?": 5, "Movement time": 5
}

annotation_dict = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "UNKNOWN": UNKNOWN}
UNKNOWN = 5
EPOCH_PER_SEC_SIZE = 30

# load files name
psgs = glob(os.path.join(raw_data_folder, "*PSG.edf"))
anns = glob(os.path.join(raw_data_folder, "*Hypnogram.edf"))
anns.sort()
psgs.sort()
anns = np.asarray(anns)
psgs = np.asarray(psgs)

for i in range(len(psgs)):
    data_file = psgs[i]
    annotation_file = anns[i]
    raw_data = read_raw_edf(data_file, preload=True, stim_channel=None) 
    samplingrate = raw_data.info['sfreq']
    raw_channel_df = raw_data.to_data_frame(scaling_time=100.0)[channel]
    raw_channel_df = raw_channel_df.to_frame()
    # read annotations from annotation files
    annotation_list = []
    with pyedflib.EdfReader(annotation_file) as reader:
        annotation = reader.readAnnotations()
        length = len(annotation[0])
        for i in range(length):
            annotation_list.append((annotation[0][i], annotation[1][i], [annotation[2][i]]))
    reader.__exit__
    
    label_index = []        
    remove_index = []    
    labels = []    
    for annotation in annotation_list:
        onset_sec, duration_sec, ann_character = annotation
        ann_string = "".join(ann_character)
        label = annotationlabel[ann_string]
        if label != UNKNOWN:
            duration_epoch = int(duration_sec / EPOCH_PER_SEC_SIZE)
            label_epoch = np.ones(duration_epoch, dtype=np.int) * label
            labels.append(label_epoch)
            idx = int(onset_sec * samplingrate) + np.arange(duration_sec * samplingrate, dtype=np.int)
            label_index.append(idx)
        else:
            idx = int(onset_sec * samplingrate) + np.arange(duration_sec * samplingrate, dtype=np.int)
            remove_index.append(idx)
    
    labels = np.hstack(labels)
    
    if len(remove_index) > 0:
        remove_index = np.hstack(remove_index)
        select_index = np.setdiff1d(np.arange(len(raw_channel_df)), remove_index)
    else:
        select_index = np.arange(len(raw_channel_df))
    
    # only data and labels will be selected
    label_index = np.hstack(label_index)
    select_index = np.intersect1d(select_index, label_index)
    
    if len(label_index) > len(select_index):
        extra_idx = np.setdiff1d(label_index, select_index)
        if np.all(extra_idx > select_index[-1]):
            trims = len(select_index) % int(EPOCH_PER_SEC_SIZE * samplingrate)
            n_label = int(math.ceil(trims / (EPOCH_PER_SEC_SIZE * samplingrate)))
            select_index = select_index[:-trims]
            labels = labels[:-n_label]
    
    raw_channel = raw_channel_df.values[select_index]
    n_epochs = len(raw_channel) / (EPOCH_PER_SEC_SIZE * samplingrate)
    
    x = np.asarray(np.split(raw_channel, n_epochs)).astype(np.float32)
    y = labels.astype(np.int32)
    
    time_windows_in_minutes = 30
    nw_index = np.where(y != annotation_dict["W"])[0]
    start_index = nw_index[0] - (time_windows_in_minutes * 2)
    end_index = nw_index[-1] + (time_windows_in_minutes * 2)
    if start_index < 0: start_index = 0
    if end_index >= len(y): end_index = len(y) - 1
    select_index = np.arange(start_index, end_index+1)
    x = x[select_index]
    y = y[select_index]
    
    filename = ntpath.basename(data_file).replace("-PSG.edf", ".npz")
    results_dict = {"x": x, "y": y, "fs": samplingrate}

    np.savez(os.path.join(output_folder, filename), **results_dict)
