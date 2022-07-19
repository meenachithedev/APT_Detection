#importing neccessary libraries
from sklearn.preprocessing import StandardScaler
import math
import csv
import numpy as np
import pandas as pd
#A class for standardising data
class DataLoader():
    
    #get the data from the given csv file
    def __init__(self, filename,cols):
        dataframe = pd.read_csv(filename)
        self.data  = dataframe.get(cols).values

    #function to produce the data frames
    def data_windows_creation(self,seq_len):
        data_windows = []
        print(len(self.data))
        print(self.data.shape)
        for i in range(len(self.data)-seq_len+1):
            data_windows.append(self.data[i:i+seq_len])
        data_windows = np.array(data_windows).astype(float)  
        return data_windows
#output files 
of='/Users/sivachidambaram/Documents/MiniProject_DS/NEW/norm_ds_time.csv'
oflen='/Users/sivachidambaram/Documents/MiniProject_DS/NEW/norm_ds_len.csv'
#StandardScaler - a function which performs standardisation i.e (data-mean/variance)
object= StandardScaler()
ob=DataLoader("/Users/sivachidambaram/Documents/MiniProject_DS/NEW/USED_DS_LSTM.csv","Time");
nparr=ob.data_windows_creation(1)

scale = object.fit_transform(nparr)
with open(of, 'w', newline='') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, lineterminator = '\n')
    for i in range(0,int(len(scale))):
        thedatawriter.writerow(scale[i])
    
object2= StandardScaler()
ob1=DataLoader("/Users/sivachidambaram/Documents/MiniProject_DS/NEW/USED_DS_LSTM.csv","Length");
nparr_len=ob1.data_windows_creation(1)
scale_len = object2.fit_transform(nparr_len)
with open(oflen, 'w', newline='') as mycsvfile:
    thedatawriter = csv.writer(mycsvfile, lineterminator = '\n')
    for i in range(0,int(len(scale_len))):
        thedatawriter.writerow(scale_len[i])