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
from keras.layers import LeakyReLU
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras import metrics
import tensorflow
import keras.backend as K
import matplotlib

#matplotlib.use('Agg')

leaky_relu_alpha = 0.1



from sklearn.model_selection import train_test_split

'''Reading data from text files'''

with open("/work/yverma/DNN_SLQ/WW_class.txt") as ww:
    WW = [line.split() for line in ww]


with open("/work/yverma/DNN_SLQ/WZ_class.txt") as wz:
    WZ = [line.split() for line in wz]


with open("/work/yverma/DNN_SLQ/ZZ_class.txt") as zz:
    ZZ = [line.split() for line in zz]


with open("/work/yverma/DNN_SLQ/DY_class.txt") as dy:
    DY = [line.split() for line in dy]

with open("/work/yverma/DNN_SLQ/singletop1_class.txt") as st1:
    ST1 = [line.split() for line in st1]

with open("/work/yverma/DNN_SLQ/singletop2_class.txt") as st2:
    ST2 = [line.split() for line in st2]

with open("/work/yverma/DNN_SLQ/singletop3_class.txt") as st3:
    ST3 = [line.split() for line in st3]

with open("/work/yverma/DNN_SLQ/singletop4_class.txt") as st4:
    ST4 = [line.split() for line in st4]

with open("/work/yverma/DNN_SLQ/tt_class.txt") as tt:
    TT = [line.split() for line in tt]

with open("/work/yverma/DNN_SLQ/LQ_5_class.txt") as lq50:
    lQ50 = [line.split() for line in lq50]

with open("/work/yverma/DNN_SLQ/LQ_8_class.txt") as lq80:
    lQ80 = [line.split() for line in lq80]

with open("/work/yverma/DNN_SLQ/LQ_11_class.txt") as lq110:
    lQ110 = [line.split() for line in lq110]

with open("/work/yverma/DNN_SLQ/LQ_14_class.txt") as lq140:
    lQ140 = [line.split() for line in lq140]

with open("/work/yverma/DNN_SLQ/LQ_17_class.txt") as lq170:
    lQ170 = [line.split() for line in lq170]

with open("/work/yverma/DNN_SLQ/LQ_20_class.txt") as lq200:
    lQ200 = [line.split() for line in lq200]

with open("/work/yverma/DNN_SLQ/LQ_23_class.txt") as lq230:
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
LQ50_arr = np.array(lQ50, dtype=float)
LQ80_arr = np.array(lQ80, dtype=float)
LQ110_arr = np.array(lQ110, dtype=float)
LQ140_arr = np.array(lQ140, dtype=float)
LQ170_arr = np.array(lQ170, dtype=float)
LQ200_arr = np.array(lQ200, dtype=float)
LQ230_arr = np.array(lQ230, dtype=float)

#Physical variables under consideration for input to DNN
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
LQ50 = pd.DataFrame(LQ50_arr, columns=lq_column)
LQ80 = pd.DataFrame(LQ80_arr, columns=lq_column)
LQ110 = pd.DataFrame(LQ110_arr, columns=lq_column)
LQ140 = pd.DataFrame(LQ140_arr, columns=lq_column)
LQ170 = pd.DataFrame(LQ170_arr, columns=lq_column)
LQ200 = pd.DataFrame(LQ200_arr, columns=lq_column)
LQ230 = pd.DataFrame(LQ230_arr, columns=lq_column)




frame = [LQ50,LQ80,LQ110,LQ140,LQ170,LQ200,LQ230]
LQ = pd.concat(frame)
LQ = LQ.dropna()

#Class labelling Background --> 0 , Signal--> 1
LQ_class = []
TT_class = []
ST1_class = []
ST2_class = []
ST3_class = []
ST4_class = []
DY_class = []
WW_class = []
WZ_class  = []
ZZ_class = []


#Ordered weights definition
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



weight_ww = w_ww*np.ones(len(WW),dtype=float)
weight_wz = w_wz*np.ones(len(WZ),dtype=float)
weight_zz = w_zz*np.ones(len(ZZ),dtype=float)
weight_dy = w_dy*np.ones(len(DY),dtype=float)
weight_st1 = w_st1*np.ones(len(ST_1),dtype=float)
weight_st2 = w_st2*np.ones(len(ST_2),dtype=float)
weight_st3 = w_st3*np.ones(len(ST_3),dtype=float)
weight_st4 = w_st4*np.ones(len(ST_4),dtype=float)
weight_tt = w_tt*np.ones(len(TT),dtype=float)
weight_50 = w_50*np.ones(len(LQ50),dtype=float)
weight_80 = w_80*np.ones(len(LQ80),dtype=float)
weight_110 = w_110*np.ones(len(LQ110),dtype=float)
weight_140 = w_140*np.ones(len(LQ140),dtype=float)
weight_170 = w_170*np.ones(len(LQ170),dtype=float)
weight_200 = w_200*np.ones(len(LQ200),dtype=float)
weight_230 = w_230*np.ones(len(LQ230),dtype=float)


weight_arr = np.concatenate((weight_50,weight_80,weight_110,weight_140,weight_170,weight_200,weight_230,weight_tt,weight_st1,weight_st2,weight_st3,weight_st4,weight_dy,weight_ww,weight_wz,weight_zz),axis=0)



for i in range(len(ST_1)):
    ST1_class.append(0)

for i in range(len(ST_2)):
    ST2_class.append(0)

for i in range(len(ST_3)):
    ST3_class.append(0)

for i in range(len(ST_4)):
    ST4_class.append(0)

for i in range(len(TT)):
    TT_class.append(0)

for i in range(len(DY)):
    DY_class.append(0)

for i in range(len(WW)):
    WW_class.append(0)

for i in range(len(WZ)):
    WZ_class.append(0)

for i in range(len(ZZ)):
    ZZ_class.append(0)

for i in range(len(LQ)):
    LQ_class.append(1)


final_LQ = LQ.to_numpy(dtype=float)
final_TT = TT.to_numpy(dtype=float)
final_DY = DY.to_numpy(dtype=float)
final_WW = WW.to_numpy(dtype=float)
final_WZ = WZ.to_numpy(dtype=float)
final_ZZ = ZZ.to_numpy(dtype=float)
final_ST1 = ST_1.to_numpy(dtype=float)
final_ST2 = ST_2.to_numpy(dtype=float)
final_ST3 = ST_3.to_numpy(dtype=float)
final_ST4 = ST_4.to_numpy(dtype=float)


LQ_class_arr = np.array(LQ_class)
TT_class_arr = np.array(TT_class)
DY_class_arr = np.array(DY_class)
WW_class_arr = np.array(WW_class)
WZ_class_arr = np.array(WZ_class)
ZZ_class_arr = np.array(ZZ_class)
ST1_class_arr = np.array(ST1_class)
ST2_class_arr = np.array(ST2_class)
ST3_class_arr = np.array(ST3_class)
ST4_class_arr = np.array(ST4_class)

final_data = np.concatenate((final_LQ,final_TT,final_ST1,final_ST2,final_ST3,final_ST4,final_DY,final_WW,final_WZ,final_ZZ),axis=0)
final_class = np.concatenate((LQ_class_arr,TT_class_arr,ST1_class_arr,ST2_class_arr,ST3_class_arr,ST4_class_arr,DY_class_arr,WW_class_arr,WZ_class_arr,ZZ_class_arr),axis=0)

final_class2 = np.c_[final_class,weight_arr]

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(final_data)
final_data = scaler.transform(final_data)

#Model Definition
def base_model():
    model = Sequential()
    model.add(Dense(200, input_dim=final_LQ.shape[1]))
    model.add(BatchNormalization())
    model.add(Dense(200))
    model.add(LeakyReLU(alpha=leaky_relu_alpha))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.2))
    model.add(Dense(200))
    model.add(LeakyReLU(alpha=leaky_relu_alpha))
    model.add(BatchNormalization())
    model.add(Dense(200))
    model.add(LeakyReLU(alpha=leaky_relu_alpha))
    model.add(BatchNormalization())
    model.add(Dense(2,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=[metrics.categorical_accuracy])
    return model



# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(final_data, final_class2, test_size=0.2, random_state=42)


weight_train = y_train[:,[1]]
weight_test = y_test[:,[1]]
y_test = y_test[:,[0]]
y_train = y_train[:,[0]]


weight_train = weight_train.flatten()
weight_test = weight_test.flatten()




# Converting class label to categorical variable(one-hot encoding)
y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)
final_class = np_utils.to_categorical(final_class, 2)


batch_size = 256
#Model Training
print("##############  TRAINING BINARY CLASSIFICATION MODEL WITH ORDERED WEIGHTS  ###############")
LQ_model = base_model()
es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=2, patience=80)
mc = ModelCheckpoint('/work/yverma/DNN_SLQ/Regression_class.h5', monitor='val_categorical_accuracy', mode='max', verbose=2, save_best_only=True)
history = LQ_model.fit(X_train, y_train,sample_weight=weight_train, batch_size= batch_size, epochs=150, validation_data=(X_test,y_test,weight_test), shuffle=False,callbacks=[es, mc],verbose=2)

saved_model = load_model('/work/yverma/DNN_SLQ/Regression_class.h5')

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


df_cm = pd.DataFrame(cm,index = [i for i in ["Background","LQ"]],columns = [i for i in ["Background","LQ"]])
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 12})# font size
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.savefig('/work/yverma/DNN_SLQ/Conf_mat.png')
plt.clf()
plt.close()



#Loss and Accuracy history plots

print(history.history.keys())
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

#Saving the predicted results into text files
class_50 = []
class_80 = []
class_110 = []
class_140 = []
class_170 = []
class_200 = []
class_230 = []
class_tt = []
class_dy = []
class_st1 = []
class_st2 = []
class_st3 = []
class_st4 = []
class_ww = []
class_wz = []
class_zz = []


test_Predict = saved_model.predict(final_data, verbose=2)

for i in range(len(final_data)):
    if weight_arr[i] == w_tt:
       class_tt.append(test_Predict[i][1])  
    if weight_arr[i] == w_dy:
       class_dy.append(test_Predict[i][1])
    if weight_arr[i] == w_st1:
       class_st1.append(test_Predict[i][1])
    if weight_arr[i] == w_st2:
       class_st2.append(test_Predict[i][1])
    if weight_arr[i] == w_st3:
       class_st3.append(test_Predict[i][1])  
    if weight_arr[i] == w_st4:
       class_st4.append(test_Predict[i][1])
    if weight_arr[i] == w_ww:
       class_ww.append(test_Predict[i][1])
    if weight_arr[i] == w_wz:
       class_wz.append(test_Predict[i][1])
    if weight_arr[i] == w_zz:
       class_zz.append(test_Predict[i][1])

    if weight_arr[i] == w_50:
       class_50.append(test_Predict[i][1])
    if weight_arr[i] == w_80:
       class_80.append(test_Predict[i][1])
    if weight_arr[i] == w_110:
       class_110.append(test_Predict[i][1])
    if weight_arr[i] == w_140:
       class_140.append(test_Predict[i][1])
    if weight_arr[i] == w_170:
       class_170.append(test_Predict[i][1])
    if weight_arr[i] == w_200:
       class_200.append(test_Predict[i][1])
    if weight_arr[i] == w_230:
       class_230.append(test_Predict[i][1])

f1 = open("/work/yverma/DNN_SLQ/LQ500_binary_ord_wet.txt", "w+")

for i in class_50:
    f1.write(str(i))
    f1.write("\n")
f1.close()

f2 = open("/work/yverma/DNN_SLQ/LQ800_binary_ord_wet.txt", "w+")

for i in class_80:
    f2.write(str(i))
    f2.write("\n")
f2.close()

f3 = open("/work/yverma/DNN_SLQ/LQ1100_binary_ord_wet.txt", "w+")

for i in class_110:
    f3.write(str(i))
    f3.write("\n")
f3.close()

f4 = open("/work/yverma/DNN_SLQ/LQ1400_binary_ord_wet.txt", "w+")

for i in class_140:
    f4.write(str(i))
    f4.write("\n")
f4.close()

f5 = open("/work/yverma/DNN_SLQ/LQ1700_binary_ord_wet.txt", "w+")

for i in class_170:
    f5.write(str(i))
    f5.write("\n")
f5.close()

f6 = open("/work/yverma/DNN_SLQ/LQ2000_binary_ord_wet.txt", "w+")

for i in class_200:
    f6.write(str(i))
    f6.write("\n")
f6.close()


f7 = open("/work/yverma/DNN_SLQ/LQ2300_binary_ord_wet.txt", "w+")

for i in class_230:
    f7.write(str(i))
    f7.write("\n")
f7.close()


f8 = open("/work/yverma/DNN_SLQ/TT_binary_ord_wet.txt", "w+")

for i in class_tt:
    f8.write(str(i))
    f8.write("\n")
f8.close()

f9 = open("/work/yverma/DNN_SLQ/ST1_binary_ord_wet.txt", "w+")

for i in class_st1:
    f9.write(str(i))
    f9.write("\n")
f9.close()

f10 = open("/work/yverma/DNN_SLQ/ST2_binary_ord_wet.txt", "w+")

for i in class_st2:
    f10.write(str(i))
    f10.write("\n")
f10.close()


f11 = open("/work/yverma/DNN_SLQ/ST3_binary_ord_wet.txt", "w+")

for i in class_st3:
    f11.write(str(i))
    f11.write("\n")
f11.close()


f12 = open("/work/yverma/DNN_SLQ/ST4_binary_ord_wet.txt", "w+")

for i in class_st4:
    f12.write(str(i))
    f12.write("\n")
f12.close()


f13 = open("/work/yverma/DNN_SLQ/DY_binary_ord_wet.txt", "w+")

for i in class_dy:
    f13.write(str(i))
    f13.write("\n")
f13.close()

f14 = open("/work/yverma/DNN_SLQ/WW_binary_ord_wet.txt", "w+")

for i in class_ww:
    f14.write(str(i))
    f14.write("\n")
f14.close()

f15 = open("/work/yverma/DNN_SLQ/WZ_binary_ord_wet.txt", "w+")

for i in class_wz:
    f15.write(str(i))
    f15.write("\n")
f15.close()


f16 = open("/work/yverma/DNN_SLQ/ZZ_binary_ord_wet.txt", "w+")

for i in class_zz:
    f16.write(str(i))
    f16.write("\n")
f16.close()









