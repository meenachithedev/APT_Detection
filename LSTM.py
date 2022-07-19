#importing necessary modules
import pandas as pd
import numpy as np
import tensorflow as tf
import csv
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import mse
from keras.models import Sequential,Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, LSTM,Flatten

#reading data from input file
ipfile="/Users/sivachidambaram/Documents/MiniProject_DS/NORMALISED_USED_LSTM_MDS.csv"
ip2="/Users/sivachidambaram/Documents/MiniProject_DS/TestSet_LSTM_MIXED.csv"
df=pd.read_csv(ipfile)
df2=pd.read_csv(ip2)
#Splitting training and testing data
train=df.iloc[:]
test=df2.iloc[:]

#converting panda array to numpy array
train_arr=np.array(train).astype(float)
test_arr=np.array(test).astype(float)

#separating inputs and outputs of particular packet for train data
#eg : [-0.38641741 -0.34578399] [1.]
ind=0
i=0
cnt=0
X_train, y_train=[],[]
X_test, y_test=[],[]
tn=[]
while(i<len(train_arr)):
        tn_child=[]
        tn_y_child=[]
        tn_child.append(train_arr[i][0])
        tn_child.append(train_arr[i][1])
        tn_y_child.append(train_arr[i][2])
        tn.append(tn_child)
        X_train.append(tn_child)
        y_train.append(tn_y_child)
        i=i+1
#separating inputs and outputs of particular packet for test data
tn1=[]
i=0
while(i<len(test_arr)):
        tn_child=[]
        tn_y_child=[]
        tn_child.append(test_arr[i][0])
        tn_child.append(test_arr[i][1])
        tn_y_child.append(test_arr[i][2])
        tn1.append(tn_child)
        X_test.append(tn_child)
        y_test.append(tn_y_child)
        i=i+1
X_test=np.array(X_test).astype(float)   
y_test=np.array(y_test).astype(float) 
X_train=np.array(X_train).astype(float)   
y_train=np.array(y_train).astype(float) 
print(X_train[0],y_train[0]," is sample train i/p")
print(X_test[0],y_test[0]," is sample test i/p")
print(train_arr.shape,"train arr shape")
print(X_train.shape, "X_train arr shape")
print(y_train.shape, "y_train arr shape")
print(X_test.shape, "X_test arr shape")
print(y_test.shape, "y_test arr shape")


#separating network flows i.e array of arrays that contains 15 packets for test and train data
X_batch_train,y_batch_train=[],[]
cnt=0
j=0
while(cnt<len(train)/15):
    tn=[]
    tn_y=[]
    for i in range(0,15):
        tn.append(X_train[j])
        tn_y.append(y_train[j])
        j=j+1
    
    X_batch_train.append(tn)
    y_batch_train.append(tn_y)
    cnt=cnt+1
cnt=0
j=0
X_batch_test,y_batch_test=[],[]
while(cnt<len(test)/15):
    tn=[]
    tn_y=[]
    for i in range(0,15):
        tn.append(X_test[j])
        tn_y.append(y_test[j])
        j=j+1
    
    X_batch_test.append(tn)
    y_batch_test.append(tn_y)
    cnt=cnt+1
X_batch_train=np.array(X_batch_train).astype(float) 
y_batch_train=np.array(y_batch_train).astype(float) 
X_batch_test=np.array(X_batch_test).astype(float) 
y_batch_test=np.array(y_batch_test).astype(float) 
print(X_batch_train.shape, "X_train batch arr shape")
print(y_batch_train.shape, "y_train batch arr shape")
print(X_batch_test.shape, "X_test batch arr shape")
print(y_batch_test.shape, "y_test batch arr shape")

#model     
n_timesteps_in=15
n_features=2
numberOfLSTMunits=20
model=Sequential()
model.add(LSTM(150,activation='sigmoid',input_shape=(n_timesteps_in,n_features),return_sequences=True,name="l1"))
model.add(Flatten(name="l2"))
#model.add(LSTM(150, name="l1"))
model.add(Dense(numberOfLSTMunits, name="l3"))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse',metrics='accuracy')
model.summary()
model.fit(X_batch_train,y_batch_train, epochs=50, batch_size=1, validation_data=(X_batch_test, y_batch_test))

#Testing model
newmodel=model.layers[:-1]
print(newmodel)
nmodel=Sequential(newmodel)
nmodel.compile(optimizer='adam', loss='mse',metrics='accuracy')
nmodel.summary()
result=nmodel.predict(X_batch_test)
result_train=nmodel.predict(X_batch_train)
print('state_h shape: after ', result.shape)
print('state_h shape: after ', result_train.shape)
opfile_fe="/Users/sivachidambaram/Documents/MiniProject_DS/lstm_extracted_features.txt"
opfile_fe2="/Users/sivachidambaram/Documents/MiniProject_DS/lstm_extracted_features_train.txt"
opf = open(opfile_fe, 'w')
opf2 = open(opfile_fe2, 'w')
k=0
filestr="/Users/sivachidambaram/Documents/MiniProject_DS/Features/f"
filestr2="/Users/sivachidambaram/Documents/MiniProject_DS/Features/ft"
for i in range(0,result.shape[1]):
    filestr+=str(i+1)+".txt";
    print(filestr);
    #print(i,"i is")
    file=open(filestr,'w')
    for j in range(0,result.shape[0]):
        #print(j,"j is")
        file.write(result[j][i].astype('str'))
        file.write("\n")
    file.close()
    filestr="/Users/sivachidambaram/Documents/MiniProject_DS/Features/f";
    
    
for i in range(0,result.shape[0]):
    print(y_test[k][0])
    opf.write(str(y_test[k][0]))
    k=k+15
    opf.write("\n")
    #print('result for the  sample/input: i \n', np.array_str(result[i]))
k=0    
for i in range(0,result_train.shape[1]):
    filestr2+=str(i+1)+".txt";
    print(filestr2);
    #print(i,"i is")
    file=open(filestr2,'w')
    for j in range(0,result_train.shape[0]):
        #print(j,"j is")
        file.write(result_train[j][i].astype('str'))
        file.write("\n")
    file.close()
    filestr2="/Users/sivachidambaram/Documents/MiniProject_DS/Features/ft";
    
    
for i in range(0,result_train.shape[0]):
    print(y_train[k][0])
    opf2.write(str(y_train[k][0]))
    k=k+15
    opf2.write("\n")
    #print('result for the  sample/input: i \n', np.array_str(result_train[i]))

