import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from glob import glob
import os
import random


TIME_WINDOW_SIZE = 100

def clip_array(x):
    x = np.clip(x / 20, -5, 5)
    return x

def load_data(path, num_of_samples = 12):
    files = sorted(glob(os.path.join(path, "*.npz")))
    data = []
    Y = []
    # randomly sampling:
    for file in files:
        npz = np.load(file)
        for _ in range(num_of_samples):
            index = random.choice(range(abs(npz['x'].shape[0]-TIME_WINDOW_SIZE)))
            x = npz['x'][index:index+TIME_WINDOW_SIZE, ...]
            y = npz['y'][index:index+TIME_WINDOW_SIZE]
            y = np.expand_dims(y, -1)
            x = clip_array(x)            
            data.append(x)
            Y.append(y)
       
    data = np.concatenate(data, axis=0)

    data = np.reshape(data, (data.shape[0],-1))

    Y = np.concatenate(Y)
    target = []
    for y in Y:
        target.append(y[0])
        
    return data, np.array(target)
    
def load_seizure_dataset(path, model_type):
  
    data, target = load_data(path)
    
   
        data = torch.from_numpy(data.astype('float32')).unsqueeze(2)
        #data = torch.from_numpy(data.astype('float32'))
        target = torch.from_numpy(target.astype('long')).long()
        dataset = TensorDataset(data, target)
   
    return dataset


