'''Author: Yogesh Verma'''
'''DeepLQ'''
# ! /usr/bin/env python

import numpy as np
import keras
import pandas as pd

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import LSTM
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import metrics
from keras.layers import Reshape
import tensorflow
import keras.backend as K
import matplotlib
import random

#matplotlib.use('Agg')
from sklearn.model_selection import train_test_split

#Reading data from text files

with open("/work/yverma/DNN_SLQ/WW_preclass.txt") as ww:
    WW = [line.split() for line in ww]


with open("/work/yverma/DNN_SLQ/WZ_preclass.txt") as wz:
    WZ = [line.split() for line in wz]


with open("/work/yverma/DNN_SLQ/ZZ_preclass.txt") as zz:
    ZZ = [line.split() for line in zz]

with open("/work/yverma/DNN_SLQ/DY_preclass.txt") as dy:
    DY = [line.split() for line in dy]

with open("/work/yverma/DNN_SLQ/singletop1_preclass.txt") as st1:
    ST1 = [line.split() for line in st1]

with open("/work/yverma/DNN_SLQ/singletop2_preclass.txt") as st2:
    ST2 = [line.split() for line in st2]

with open("/work/yverma/DNN_SLQ/singletop3_preclass.txt") as st3:
    ST3 = [line.split() for line in st3]

with open("/work/yverma/DNN_SLQ/singletop4_preclass.txt") as st4:
    ST4 = [line.split() for line in st4]

with open("/work/yverma/DNN_SLQ/tt_preclass.txt") as tt:
    TT = [line.split() for line in tt]
tt =0

with open("/work/yverma/DNN_SLQ/LQ_5_preclass.txt") as lq50:
    lQ50 = [line.split() for line in lq50]

with open("/work/yverma/DNN_SLQ/LQ_8_preclass.txt") as lq80:
    lQ80 = [line.split() for line in lq80]

with open("/work/yverma/DNN_SLQ/LQ_11_preclass.txt") as lq110:
    lQ110 = [line.split() for line in lq110]

with open("/work/yverma/DNN_SLQ/LQ_14_preclass.txt") as lq140:
    lQ140 = [line.split() for line in lq140]

with open("/work/yverma/DNN_SLQ/LQ_17_preclass.txt") as lq170:
    lQ170 = [line.split() for line in lq170]

with open("/work/yverma/DNN_SLQ/LQ_20_preclass.txt") as lq200:
    lQ200 = [line.split() for line in lq200]

with open("/work/yverma/DNN_SLQ/LQ_23_preclass.txt") as lq230:
    lQ230 = [line.split() for line in lq230]




WW_arr =  np.array(WW, dtype=float)
WZ_arr =  np.array(WZ, dtype=float)
ZZ_arr =  np.array(ZZ, dtype=float)
DY_arr =  np.array(DY, dtype=float)
ST1_arr = np.array(ST1, dtype=float)
ST2_arr = np.array(ST2, dtype=float)
ST3_arr = np.array(ST3, dtype=float)
ST4_arr = np.array(ST4, dtype=float)
TT_arr = np.array(TT, dtype=float)
TT = 0
LQ50_arr = np.array(lQ50, dtype=float)
LQ80_arr = np.array(lQ80, dtype=float)
LQ110_arr = np.array(lQ110, dtype=float)
LQ140_arr = np.array(lQ140, dtype=float)
LQ170_arr = np.array(lQ170, dtype=float)
LQ200_arr = np.array(lQ200, dtype=float)
LQ230_arr = np.array(lQ230, dtype=float)



lq_column = ["mass_jet1", "mass_jet2","mass_jet3","mass_jet4", "Jet1_pt", "Jet2_pt","Jet3_pt","Jet4_pt", "Jet1_eta", "Jet2_eta",
             "Jet3_eta","Jet4_eta","Jet1_phi", "Jet2_phi","Jet3_phi","Jet4_phi", "B_Scaore_J1", "B_Score_J2","B_Score_J3","B_Score_J4", "MET", "MET_phi", "ST",
             "ST_MET", "tau1_Mvis", "tau2_Mvis", "tau1_phivis", "tau2_phivis", "tau1_ptvis", "tau2_ptvis",
             "tau1_etavis", "tau2_etavis","pTc1","pTc2","mvisc1","mvisc2","etac1","etac2","phic1","phic2"]




WW = pd.DataFrame(WW_arr, columns=lq_column)
WZ = pd.DataFrame(WZ_arr, columns=lq_column)
ZZ = pd.DataFrame(ZZ_arr, columns=lq_column)
DY = pd.DataFrame(DY_arr, columns=lq_column)
ST_1 = pd.DataFrame(ST1_arr, columns=lq_column)
ST_2 = pd.DataFrame(ST2_arr, columns=lq_column)
ST_3 = pd.DataFrame(ST3_arr, columns=lq_column)
ST_4 = pd.DataFrame(ST4_arr, columns=lq_column)
TT = pd.DataFrame(TT_arr, columns=lq_column)
TT_arr = 0
LQ50 = pd.DataFrame(LQ50_arr, columns=lq_column)
LQ80 = pd.DataFrame(LQ80_arr, columns=lq_column)
LQ110 = pd.DataFrame(LQ110_arr, columns=lq_column)
LQ140 = pd.DataFrame(LQ140_arr, columns=lq_column)
LQ170 = pd.DataFrame(LQ170_arr, columns=lq_column)
LQ200 = pd.DataFrame(LQ200_arr, columns=lq_column)
LQ230 = pd.DataFrame(LQ230_arr, columns=lq_column)


#Ordered weight initialization w_i*n_i  = w_j*n_j

w_tt = 1
w_dy = len(TT)*w_tt/len(DY)
w_ww = len(TT)*w_tt/len(WW)
w_wz = len(TT)*w_tt/len(WZ)
w_zz = len(TT)*w_tt/len(ZZ)
w_st1 = len(TT)*w_tt/len(ST_1)
w_st2 = len(TT)*w_tt/len(ST_2)
w_st3 = len(TT)*w_tt/len(ST_3)
w_st4 = len(TT)*w_tt/len(ST_4)
w_50 = len(TT)*w_tt/len(LQ50)
w_80 = len(TT)*w_tt/len(LQ80)
w_110 = len(TT)*w_tt/len(LQ110)
w_140 = len(TT)*w_tt/len(LQ140)
w_170 = len(TT)*w_tt/len(LQ170)
w_200 = len(TT)*w_tt/len(LQ200)
w_230 = len(TT)*w_tt/len(LQ230)


weight_dy = w_dy*np.ones(len(DY),dtype=float)
weight_st1 = w_st1*np.ones(len(ST_1),dtype=float)
weight_st2 = w_st2*np.ones(len(ST_2),dtype=float)
weight_st3 = w_st3*np.ones(len(ST_3),dtype=float)
weight_st4 = w_st4*np.ones(len(ST_4),dtype=float)
weight_ww = w_ww*np.ones(len(WW),dtype=float)
weight_wz = w_wz*np.ones(len(WZ),dtype=float)
weight_zz = w_zz*np.ones(len(ZZ),dtype=float)
weight_tt = w_tt*np.ones(len(TT),dtype=float)
weight_50 = w_50*np.ones(len(LQ50),dtype=float)
weight_80 = w_80*np.ones(len(LQ80),dtype=float)
weight_110 = w_110*np.ones(len(LQ110),dtype=float)
weight_140 = w_140*np.ones(len(LQ140),dtype=float)
weight_170 = w_170*np.ones(len(LQ170),dtype=float)
weight_200 = w_200*np.ones(len(LQ200),dtype=float)
weight_230 = w_230*np.ones(len(LQ230),dtype=float)




#MASS Regression Model Definition


def mass_model():
    model = Sequential()
    model.add(Dense(len(lq_column2), input_dim=len(lq_column2)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model





#Mass Label

mass_50 = 500*np.ones(len(LQ50),dtype=float)
mass_80 = 800*np.ones(len(LQ80),dtype=float)
mass_110 = 1100*np.ones(len(LQ110),dtype=float)
mass_140 = 1400*np.ones(len(LQ140),dtype=float)
mass_170 = 1700*np.ones(len(LQ170),dtype=float)
mass_200 = 2000*np.ones(len(LQ200),dtype=float)
mass_230 = 2300*np.ones(len(LQ230),dtype=float)


mass_arr = np.concatenate((mass_50,mass_80,mass_110,mass_140,mass_170,mass_200,mass_230),axis=0)


frame_mass = [LQ50,LQ80,LQ110,LQ140,LQ170,LQ200,LQ230]
LQ_mass = pd.concat(frame_mass)
LQ_mass = LQ_mass.dropna()


final_LQ_mass = LQ_mass.to_numpy(dtype=float)
finalTT = TT.to_numpy(dtype=float)
finalST1 = ST_1.to_numpy(dtype=float)
finalST2 = ST_2.to_numpy(dtype=float)
finalST3 = ST_3.to_numpy(dtype=float)
finalST4 = ST_4.to_numpy(dtype=float)
finalDY = DY.to_numpy(dtype=float)
finalWW = WW.to_numpy(dtype=float)
finalWZ = WZ.to_numpy(dtype=float)
finalZZ = ZZ.to_numpy(dtype=float)

#Scaling
scaler_mass = MinMaxScaler(feature_range=(0,1))
scaler_mass.fit(final_LQ_mass)
final_LQ_mass = scaler_mass.transform(final_LQ_mass)
finalTT = scaler_mass.transform(finalTT)
finalTST1 = scaler_mass.transform(finalST1)
finalST2 = scaler_mass.transform(finalST2)
finalST3 = scaler_mass.transform(finalST3)
finalST4 = scaler_mass.transform(finalST4)
finalDY = scaler_mass.transform(finalDY)
finalWW = scaler_mass.transform(finalWW)
finalWZ = scaler_mass.transform(finalWZ)
finalZZ = scaler_mass.transform(finalZZ)


print(final_LQ_mass.shape,finalTT.shape)


data_train, data_test, mass_train, mass_test = train_test_split(final_LQ_mass, mass_arr, test_size=0.1, random_state=32)
batch_size = 256



print("####  TRAINING THE MASS MODEL OF REGRESSION TO RECONSTRUCT LQ MASS  #####")

LQ_mass_model = mass_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=800)
mc = ModelCheckpoint('/work/yverma/DNN_SLQ/Regression_multi_mass.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
LQ_mass_model = LQ_mass_model.fit(data_train, mass_train, batch_size= batch_size, epochs=100, validation_data=(data_test,mass_test), shuffle=False,callbacks=[es, mc],verbose=2)


saved_model_mass = load_model('/work/yverma/DNN_SLQ/Regression_multi_mass.h5')

tt_Predict = saved_model_mass.predict(finalTT)
finalTT = 0
st1_Predict = saved_model_mass.predict(finalST1)
st2_Predict = saved_model_mass.predict(finalST2)
st3_Predict = saved_model_mass.predict(finalST3)
st4_Predict = saved_model_mass.predict(finalST4)
dy_Predict = saved_model_mass.predict(finalDY)
ww_Predict = saved_model_mass.predict(finalWW)
wz_Predict = saved_model_mass.predict(finalWZ)
zz_Predict = saved_model_mass.predict(finalZZ)




lq_Predict = saved_model_mass.predict(final_LQ_mass)

#Prediction of mass of signal and background to be used in DNN Classification

LQ_mass = []
TT_mass = []
ST1_mass= []
ST2_mass = []
ST3_mass= []
ST4_mass = []
DY_mass = []
WW_mass=[]
WZ_mass = []
ZZ_mass = []



for i in range(len(lq_Predict)):
    LQ_mass.append(lq_Predict[i][0])


for i in range(len(tt_Predict)):
    TT_mass.append(tt_Predict[i][0])
tt_Predict = 0

for i in range(len(st1_Predict)):
    ST1_mass.append(st1_Predict[i][0])

for i in range(len(st2_Predict)):
    ST2_mass.append(st2_Predict[i][0])

for i in range(len(st3_Predict)):
    ST3_mass.append(st3_Predict[i][0])

for i in range(len(st4_Predict)):
    ST4_mass.append(st4_Predict[i][0])


for i in range(len(dy_Predict)):
    DY_mass.append(dy_Predict[i][0])

for i in range(len(ww_Predict)):
    WW_mass.append(ww_Predict[i][0])


for i in range(len(wz_Predict)):
    WZ_mass.append(wz_Predict[i][0])


for i in range(len(zz_Predict)):
    ZZ_mass.append(zz_Predict[i][0])



#Weight training array

weight_arr = np.concatenate((weight_50,weight_80,weight_110,weight_140,weight_170,weight_200,weight_230,weight_tt,weight_dy,weight_st1,weight_st2,weight_st3,weight_st4,weight_ww,weight_wz,weight_zz),axis=0)


#Class definition


LQ50_class = []
LQ80_class = []
LQ110_class = []
LQ140_class = []
LQ170_class = []
LQ200_class = []
LQ230_class = []
BKG_class = []

for i in range(len(TT)+len(ST_1)+len(ST_2) + len(ST_3)+len(ST_4)+len(WW)+len(DY)+len(WZ)+len(ZZ)):
    BKG_class.append(0)

for i in range(len(LQ50)):
    LQ50_class.append(1)

for i in range(len(LQ80)):
    LQ80_class.append(2)

for i in range(len(LQ110)):
    LQ110_class.append(3)

for i in range(len(LQ140)):
    LQ140_class.append(4)

for i in range(len(LQ170)):
    LQ170_class.append(5)

for i in range(len(LQ200)):
    LQ200_class.append(6)

for i in range(len(LQ230)):
    LQ230_class.append(7)



LQ50['class'] = LQ50_class
LQ80['class'] = LQ80_class
LQ110['class'] = LQ110_class
LQ140['class'] = LQ140_class
LQ170['class'] = LQ170_class
LQ200['class'] = LQ200_class
LQ230['class'] = LQ230_class


frame = [LQ50,LQ80,LQ110,LQ140,LQ170,LQ200,LQ230]
LQ = pd.concat(frame)
LQ = LQ.dropna()

LQ['mass']  = LQ_mass
TT['mass'] = TT_mass
DY['mass'] = DY_mass
ST_1['mass'] = ST1_mass
ST_2['mass'] = ST2_mass
ST_3['mass'] = ST3_mass
ST_4['mass'] = ST4_mass
WW['mass'] = WW_mass
WZ['mass'] = WZ_mass
ZZ['mass'] = ZZ_mass

TT_mass = 0

bkg_frame = [TT,DY,ST_1,ST_2,ST_3,ST_4,WW,WZ,ZZ]
BKG = pd.concat(bkg_frame)
BKG = BKG.dropna()


LQ_class = LQ['class'].to_numpy(dtype=float)
LQ = LQ.drop('class', axis=1)


final_LQ = LQ.to_numpy(dtype=float)
final_BKG = BKG.to_numpy(dtype=float)
BKG = 0 





LQ_class_arr = np.array(LQ_class)
BKG_class_arr = np.array(BKG_class)
BKG_class = 0


final_data = np.concatenate((final_LQ,final_BKG),axis=0)
final_BKG = 0
final_class = np.concatenate((LQ_class_arr,BKG_class_arr),axis=0)
BKG_class_arr = 0
final_class2 = np.c_[final_class,weight_arr]


#Scaling

scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(final_data)
final_data = scaler.transform(final_data)


#Classification Model definition

def base_model():
    model = Sequential()
    model.add(Dense(200, input_dim=final_LQ.shape[1]))
    model.add(BatchNormalization())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(rate = 0.1))
    model.add(Dense(200, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(200, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(8,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[metrics.categorical_accuracy])
    return model






X_train, X_test, y_train, y_test = train_test_split(final_data, final_class2, test_size=0.1, random_state=32)

weight_train = y_train[:,[1]]
weight_test = y_test[:,[1]]
y_test = y_test[:,[0]]
y_train = y_train[:,[0]]


weight_train = weight_train.flatten()
weight_test = weight_test.flatten()


y_train = np_utils.to_categorical(y_train, 8)
y_test = np_utils.to_categorical(y_test, 8)
final_class = np_utils.to_categorical(final_class, 8)


batch_size = 256



print("TRAINING THE Classificaton MODEL")

LQ_model = base_model()
es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=2, patience=800)
mc = ModelCheckpoint('/work/yverma/DNN_SLQ/Regression_multi_class.h5', monitor='val_categorical_accuracy', mode='max', verbose=2, save_best_only=True)
history = LQ_model.fit(X_train, y_train,sample_weight=weight_train, batch_size= batch_size, epochs=100, validation_data=(X_test,y_test,weight_test), shuffle=False,callbacks=[es, mc],verbose=2)

saved_model = load_model('/work/yverma/DNN_SLQ/Regression_multi_class.h5')

scores_train = saved_model.evaluate(X_train, y_train, verbose=0)
scores_test  = saved_model.evaluate(X_test,  y_test,  verbose=0)
print("Accuracy Train: %.2f%% , Test: %.2f%% " % (scores_train[1]*100, scores_test[1]*100)) 


YY_pred = saved_model.predict(X_test, verbose=2)
yy_pred = np.argmax(YY_pred, axis=1)

yy_test2 = np.argmax(y_test, axis=1)

cm = confusion_matrix(np.argmax(y_test,axis=1),yy_pred)                         #Confusion matrix


# Visualizing of confusion matrix
import seaborn as sn
import pandas  as pd


df_cm = pd.DataFrame(cm,index = [i for i in ["BKG","LQ(500)","LQ(800)","LQ(1100)","LQ(1400)","LQ(1700)","LQ(2000)","LQ(2300)"]],columns = [i for i in ["BKG","LQ(500)","LQ(800)","LQ(1100)","LQ(1400)","LQ(1700)","LQ(2000)","LQ(2300)"]])
plt.figure(figsize = (18,15))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 15})# font size
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig('/work/yverma/DNN_SLQ/Conf_mat.png')
plt.clf()
plt.close()

YY_pred = 0
yy_pred= 0
yy_test2 = 0



#Plots of Loss and accuracy


plt.figure(figsize=(10,10))
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('/work/yverma/DNN_SLQ/acc_class.png')
plt.clf()
plt.close()


plt.figure(figsize=(15,12))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('/work/yverma/DNN_SLQ/loss_class.png')
plt.clf()
plt.close()



class_50 = []
class_80 = []
class_110 = []
class_140 = []
class_170 = []
class_200 = []
class_230 = []
class_tt = []


test_Predict = saved_model.predict(final_data, verbose=2)
Ttp = np.argmax(final_class, axis=1)

for i in range(len(Ttp)):
    if Ttp[i] == 0:
       class_tt.append(test_Predict[i][0])
    if Ttp[i] == 1:
       class_50.append(test_Predict[i][1])
    if Ttp[i] == 2:
       class_80.append(test_Predict[i][2])
    if Ttp[i] == 3:
       class_110.append(test_Predict[i][3])
    if Ttp[i] == 4:
       class_140.append(test_Predict[i][4])
    if Ttp[i] == 5:
       class_170.append(test_Predict[i][5])
    if Ttp[i] == 6:
       class_200.append(test_Predict[i][6])
    if Ttp[i] == 7:
       class_230.append(test_Predict[i][7])
      
      

nodes = []

for i in range(8):
    t1 = []
    t2 = []
    t3 = []
    t4  = []
    t5 = []
    t6 = []
    t7 = []
    t8 = []
    for j in range(len(Ttp)):
        if Ttp[j] == i:
        	t1.append(test_Predict[j][0])
        	t2.append(test_Predict[j][1])
        	t3.append(test_Predict[j][2])
        	t4.append(test_Predict[j][3])
        	t5.append(test_Predict[j][4])
        	t6.append(test_Predict[j][5])
        	t7.append(test_Predict[j][6])
        	t8.append(test_Predict[j][7])
    nodes.append([t1,t2,t3,t4,t5,t6,t7,t8])

#Saving into files the output distribution


f1 = open("/work/yverma/DNN_SLQ/LQ500_multi.txt", "w+")
f2 = open("/work/yverma/DNN_SLQ/LQ800_multi.txt", "w+")
f3 = open("/work/yverma/DNN_SLQ/LQ1100_multi.txt", "w+")
f4 = open("/work/yverma/DNN_SLQ/LQ1400_multi.txt", "w+")
f5 = open("/work/yverma/DNN_SLQ/LQ1700_multi.txt", "w+")
f6 = open("/work/yverma/DNN_SLQ/LQ2000_multi.txt", "w+")
f7 = open("/work/yverma/DNN_SLQ/LQ2300_multi.txt", "w+")
f8 = open("/work/yverma/DNN_SLQ/TT_multi.txt", "w+")
f9 = open("/work/yverma/DNN_SLQ/DY_multi.txt", "w+")
f10 = open("/work/yverma/DNN_SLQ/ST1_multi.txt", "w+")
f11 = open("/work/yverma/DNN_SLQ/ST2_multi.txt", "w+")
f12 = open("/work/yverma/DNN_SLQ/ST3_multi.txt", "w+")
f13 = open("/work/yverma/DNN_SLQ/ST4_multi.txt", "w+")
f14 = open("/work/yverma/DNN_SLQ/WW_multi.txt", "w+")
f15 = open("/work/yverma/DNN_SLQ/WZ_multi.txt", "w+")
f16 = open("/work/yverma/DNN_SLQ/ZZ_multi.txt", "w+")



for i in range(len(Ttp)):
    if Ttp[i] == 0:
     if weight_arr[i] == sigma1:
       f8.write(str(test_Predict[i][0]))
       f8.write("\t")
       f8.write(str(test_Predict[i][1]))
       f8.write("\t")
       f8.write(str(test_Predict[i][2]))
       f8.write("\t")
       f8.write(str(test_Predict[i][3]))
       f8.write("\t")
       f8.write(str(test_Predict[i][4]))
       f8.write("\t")
       f8.write(str(test_Predict[i][5]))
       f8.write("\t")
       f8.write(str(test_Predict[i][6]))
       f8.write("\t")
       f8.write(str(test_Predict[i][7]))
       f8.write("\n")

     if weight_arr[i] == wdy:
       f9.write(str(test_Predict[i][0]))
       f9.write("\t")
       f9.write(str(test_Predict[i][1]))
       f9.write("\t")
       f9.write(str(test_Predict[i][2]))
       f9.write("\t")
       f9.write(str(test_Predict[i][3]))
       f9.write("\t")
       f9.write(str(test_Predict[i][4]))
       f9.write("\t")
       f9.write(str(test_Predict[i][5]))
       f9.write("\t")
       f9.write(str(test_Predict[i][6]))
       f9.write("\t")
       f9.write(str(test_Predict[i][7]))
       f9.write("\n")

     if weight_arr[i] == wst1:
       f10.write(str(test_Predict[i][0]))
       f10.write("\t")
       f10.write(str(test_Predict[i][1]))
       f10.write("\t")
       f10.write(str(test_Predict[i][2]))
       f10.write("\t")
       f10.write(str(test_Predict[i][3]))
       f10.write("\t")
       f10.write(str(test_Predict[i][4]))
       f10.write("\t")
       f10.write(str(test_Predict[i][5]))
       f10.write("\t")
       f10.write(str(test_Predict[i][6]))
       f10.write("\t")
       f10.write(str(test_Predict[i][7]))
       f10.write("\n")


     if weight_arr[i] == wst2:
       f11.write(str(test_Predict[i][0]))
       f11.write("\t")
       f11.write(str(test_Predict[i][1]))
       f11.write("\t")
       f11.write(str(test_Predict[i][2]))
       f11.write("\t")
       f11.write(str(test_Predict[i][3]))
       f11.write("\t")
       f11.write(str(test_Predict[i][4]))
       f11.write("\t")
       f11.write(str(test_Predict[i][5]))
       f11.write("\t")
       f11.write(str(test_Predict[i][6]))
       f11.write("\t")
       f11.write(str(test_Predict[i][7]))
       f11.write("\n")

     if weight_arr[i] == wst3:
       f12.write(str(test_Predict[i][0]))
       f12.write("\t")
       f12.write(str(test_Predict[i][1]))
       f12.write("\t")
       f12.write(str(test_Predict[i][2]))
       f12.write("\t")
       f12.write(str(test_Predict[i][3]))
       f12.write("\t")
       f12.write(str(test_Predict[i][4]))
       f12.write("\t")
       f12.write(str(test_Predict[i][5]))
       f12.write("\t")
       f12.write(str(test_Predict[i][6]))
       f12.write("\t")
       f12.write(str(test_Predict[i][7]))
       f12.write("\n")

     if weight_arr[i] == wst4:
       f13.write(str(test_Predict[i][0]))
       f13.write("\t")
       f13.write(str(test_Predict[i][1]))
       f13.write("\t")
       f13.write(str(test_Predict[i][2]))
       f13.write("\t")
       f13.write(str(test_Predict[i][3]))
       f13.write("\t")
       f13.write(str(test_Predict[i][4]))
       f13.write("\t")
       f13.write(str(test_Predict[i][5]))
       f13.write("\t")
       f13.write(str(test_Predict[i][6]))
       f13.write("\t")
       f13.write(str(test_Predict[i][7]))
       f13.write("\n")


     if weight_arr[i] == www:
       f14.write(str(test_Predict[i][0]))
       f14.write("\t")
       f14.write(str(test_Predict[i][1]))
       f14.write("\t")
       f14.write(str(test_Predict[i][2]))
       f14.write("\t")
       f14.write(str(test_Predict[i][3]))
       f14.write("\t")
       f14.write(str(test_Predict[i][4]))
       f14.write("\t")
       f14.write(str(test_Predict[i][5]))
       f14.write("\t")
       f14.write(str(test_Predict[i][6]))
       f14.write("\t")
       f14.write(str(test_Predict[i][7]))
       f14.write("\n")

     if weight_arr[i] == wwz:
       f15.write(str(test_Predict[i][0]))
       f15.write("\t")
       f15.write(str(test_Predict[i][1]))
       f15.write("\t")
       f15.write(str(test_Predict[i][2]))
       f15.write("\t")
       f15.write(str(test_Predict[i][3]))
       f15.write("\t")
       f15.write(str(test_Predict[i][4]))
       f15.write("\t")
       f15.write(str(test_Predict[i][5]))
       f15.write("\t")
       f15.write(str(test_Predict[i][6]))
       f15.write("\t")
       f15.write(str(test_Predict[i][7]))
       f15.write("\n")


     if weight_arr[i] == wzz:
       f16.write(str(test_Predict[i][0]))
       f16.write("\t")
       f16.write(str(test_Predict[i][1]))
       f16.write("\t")
       f16.write(str(test_Predict[i][2]))
       f16.write("\t")
       f16.write(str(test_Predict[i][3]))
       f16.write("\t")
       f16.write(str(test_Predict[i][4]))
       f16.write("\t")
       f16.write(str(test_Predict[i][5]))
       f16.write("\t")
       f16.write(str(test_Predict[i][6]))
       f16.write("\t")
       f16.write(str(test_Predict[i][7]))
       f16.write("\n")



    if Ttp[i] == 1:
       f1.write(str(test_Predict[i][0]))
       f1.write("\t")
       f1.write(str(test_Predict[i][1]))
       f1.write("\t")
       f1.write(str(test_Predict[i][2]))
       f1.write("\t")
       f1.write(str(test_Predict[i][3]))
       f1.write("\t")
       f1.write(str(test_Predict[i][4]))
       f1.write("\t")
       f1.write(str(test_Predict[i][5]))
       f1.write("\t")
       f1.write(str(test_Predict[i][6]))
       f1.write("\t")
       f1.write(str(test_Predict[i][7]))
       f1.write("\n")
    if Ttp[i] == 2:
       f2.write(str(test_Predict[i][0]))
       f2.write("\t")
       f2.write(str(test_Predict[i][1]))
       f2.write("\t")
       f2.write(str(test_Predict[i][2]))
       f2.write("\t")
       f2.write(str(test_Predict[i][3]))
       f2.write("\t")
       f2.write(str(test_Predict[i][4]))
       f2.write("\t")
       f2.write(str(test_Predict[i][5]))
       f2.write("\t")
       f2.write(str(test_Predict[i][6]))
       f2.write("\t")
       f2.write(str(test_Predict[i][7]))
       f2.write("\n")
    if Ttp[i] == 3:
       f3.write(str(test_Predict[i][0]))
       f3.write("\t")
       f3.write(str(test_Predict[i][1]))
       f3.write("\t")
       f3.write(str(test_Predict[i][2]))
       f3.write("\t")
       f3.write(str(test_Predict[i][3]))
       f3.write("\t")
       f3.write(str(test_Predict[i][4]))
       f3.write("\t")
       f3.write(str(test_Predict[i][5]))
       f3.write("\t")
       f3.write(str(test_Predict[i][6]))
       f3.write("\t")
       f3.write(str(test_Predict[i][7]))
       f3.write("\n")
    if Ttp[i] == 4:
       f4.write(str(test_Predict[i][0]))
       f4.write("\t")
       f4.write(str(test_Predict[i][1]))
       f4.write("\t")
       f4.write(str(test_Predict[i][2]))
       f4.write("\t")
       f4.write(str(test_Predict[i][3]))
       f4.write("\t")
       f4.write(str(test_Predict[i][4]))
       f4.write("\t")
       f4.write(str(test_Predict[i][5]))
       f4.write("\t")
       f4.write(str(test_Predict[i][6]))
       f4.write("\t")
       f4.write(str(test_Predict[i][7]))
       f4.write("\n")
    if Ttp[i] == 5:
       f5.write(str(test_Predict[i][0]))
       f5.write("\t")
       f5.write(str(test_Predict[i][1]))
       f5.write("\t")
       f5.write(str(test_Predict[i][2]))
       f5.write("\t")
       f5.write(str(test_Predict[i][3]))
       f5.write("\t")
       f5.write(str(test_Predict[i][4]))
       f5.write("\t")
       f5.write(str(test_Predict[i][5]))
       f5.write("\t")
       f5.write(str(test_Predict[i][6]))
       f5.write("\t")
       f5.write(str(test_Predict[i][7]))
       f5.write("\n")
    if Ttp[i] == 6:
       f6.write(str(test_Predict[i][0]))
       f6.write("\t")
       f6.write(str(test_Predict[i][1]))
       f6.write("\t")
       f6.write(str(test_Predict[i][2]))
       f6.write("\t")
       f6.write(str(test_Predict[i][3]))
       f6.write("\t")
       f6.write(str(test_Predict[i][4]))
       f6.write("\t")
       f6.write(str(test_Predict[i][5]))
       f6.write("\t")
       f6.write(str(test_Predict[i][6]))
       f6.write("\t")
       f6.write(str(test_Predict[i][7]))
       f6.write("\n")
    if Ttp[i] == 7:
       f7.write(str(test_Predict[i][0]))
       f7.write("\t")
       f7.write(str(test_Predict[i][1]))
       f7.write("\t")
       f7.write(str(test_Predict[i][2]))
       f7.write("\t")
       f7.write(str(test_Predict[i][3]))
       f7.write("\t")
       f7.write(str(test_Predict[i][4]))
       f7.write("\t")
       f7.write(str(test_Predict[i][5]))
       f7.write("\t")
       f7.write(str(test_Predict[i][6]))
       f7.write("\t")
       f7.write(str(test_Predict[i][7]))
       f7.write("\n")
              
f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()
f7.close()
f8.close()
f9.close()
f10.close()
f11.close()
f12.close()
f13.close()
f14.close()
f15.close()
f16.close()






























