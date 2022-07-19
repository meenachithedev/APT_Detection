import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D, ZeroPadding2D, Activation
from keras.layers import Conv1D, ZeroPadding1D, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D

train = pd.read_csv("C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn_norm_train.csv").values
test  = pd.read_csv("C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn_norm_test.csv").values
print(train.shape)
print("train",train[0])

trainX = train[:,:8].reshape(train.shape[0],1,8).astype( 'float32' )
print(trainX.shape)
print(trainX[0])

y_train = train[:,8:]
y_train = y_train[:,:].reshape(y_train.shape[0],1,1).astype( 'float32' )
print(y_train.shape)
print(y_train[0])

testX = test[:,:8].reshape(test.shape[0],1,8).astype( 'float32' )
print(testX.shape)
print(testX[0])

y_test = test[:,8:]
y_test = y_test[:,:].reshape(y_test.shape[0],1,1).astype( 'float32' )
print(y_test.shape)
print(y_test[0])

model=Sequential()
#conv1 layer
model.add(Convolution1D(32,2, padding= 'same' , input_shape=(1,8),activation= 'relu',data_format='channels_last' ))
#pooling1 layer
model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
#conv2 layer
model.add(Convolution1D(32, 2, 2,padding="same", activation= 'relu' ))
#maxpooling2 layer
model.add(MaxPooling1D(pool_size=2,strides=2, padding='same'))
#fc layer
model.add(Flatten())
#op layer
model.add(Dense(30))
# for training 
model.add(Dense(1))
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
model.summary()
model.fit(trainX, y_train,epochs=70,batch_size= trainX.shape[0])

#testing model
newmodel=model.layers[:-1]
print(newmodel)
nmodel=Sequential(newmodel)
nmodel.compile(optimizer='adam', loss='binary_crossentropy',metrics='accuracy')
nmodel.summary()
result=nmodel.predict(testX)
result_train=nmodel.predict(trainX)
print('state_h shape: after ', result.shape)
print('state_h shape: after ', result1.shape)
print('result for the first sample/input: \n', result[0],'\n',result[1],'\n',result[2])
opfile_fe="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\cnn_extracted_features.txt"
opfile_fe_t="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\train\\cnn_extracted_features1.txt"
opf = open(opfile_fe, 'w')
opf_t = open(opfile_fe_t, 'w')
k=0
filestr="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\f"
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
    filestr="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\f"
    
for i in range(0,result.shape[0]):
    a=int(y_test[k][0][0])
    #print(a)
    opf.write(str(a))
    k=k+1
    opf.write("\n")
    #print('result for the  sample/input: i \n', np.array_str(result[i]))

#print('result for the first sample/input: \n', result[0],result[1])
#score = nmodel.evaluate(testX, y_test, batch_size=testX.shape[0])
#print('test score ',score)

filestr1="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\train\\feature"
for i in range(0,result1.shape[1]):
    filestr1+=str(i+1)+".txt";
    print(filestr1);
    #print(i,"i is")
    file1=open(filestr1,'w')
    for j in range(0,result1.shape[0]):
        #print(j,"j is")
        file1.write(result1[j][i].astype('str'))
        file1.write("\n")
    file1.close()
    filestr1="C:\\Users\\meena\\Documents\\MiniProjectAbstractsub\\code\\cnn\\train\\feature";
k=0    
for i in range(0,result1.shape[0]):
    a=int(y_train[k][0][0])
    #print(a)
    opf_t.write(str(a))
    k=k+1
    opf_t.write("\n")