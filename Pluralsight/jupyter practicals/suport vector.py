# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:28:06 2017

@author: 473703
"""

import numpy as np
import pandas as pd


datafile='/Users/473703/Documents/ML/Pluralsight/jupyter practicals/ad.data'
data=pd.read_csv(datafile,sep=",",header=None,low_memory=False)
data.head(20)

# Check whether a given value is missing value, if yes chnage it to NaN
def toNum(cell):
    try:
        return np.float(cell)
    except:
        return np.nan
    
# Apply Missing value check to a column /Padas series
def seriestoNum(series):
    return series.apply(toNum)
# traning Data ready
train_data=data.iloc[:,0:-1].apply(seriestoNum)
train_data.head(20)

# Traning label ready

def toLabel(str):
    if str=='ad.':
        return 1
    else:
        return 0
    
train_labels=data.iloc[train_data.index,-1].apply(toLabel)
train_labels  

# Traning phase  