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
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import tensorflow
import keras.backend as K
import matplotlib
from sklearn.model_selection import train_test_split

#weights corresponding to each event
sigma1 = 1.49365004749e-05
sigma2 = 1.93210659898e-05
sigma3 = 8.57084357084e-07
sigma4 = 7.67977184763e-08
sigma5 = 9.46464646465e-09
sigma6 = 1.41248720573e-09
sigma7 = 2.28928199792e-10
sigma8 = 3.89424062308e-11
wst1 = 0.00001093554
wst2 = 0.00001620925
wst3 = 0.00000455268
wst4 = 0.00000427091
wdy = 0.00013864453
wwz = 0.0000276
www = 0.0000763371
wzz = 0.00001226183



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



#Weight used to segregate events into various signals to write in files
weight_50 = sigma2*np.ones(len(LQ50_arr),dtype=float)
weight_80 = sigma3*np.ones(len(LQ80_arr),dtype=float)
weight_110 = sigma4*np.ones(len(LQ110_arr),dtype=float)
weight_140 = sigma5*np.ones(len(LQ140_arr),dtype=float)
weight_170 = sigma6*np.ones(len(LQ170_arr),dtype=float)
weight_200 = sigma7*np.ones(len(LQ200_arr),dtype=float)
weight_230 = sigma8*np.ones(len(LQ230_arr),dtype=float)

weight_arr = np.concatenate((weight_50,weight_80,weight_110,weight_140,weight_170,weight_200,weight_230),axis=0)

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



#Targeted output variable mLQ of event

mass_50 = 500*np.ones(len(LQ50),dtype=float)
mass_80 = 800*np.ones(len(LQ80),dtype=float)
mass_110 = 1100*np.ones(len(LQ110),dtype=float)
mass_140 = 1400*np.ones(len(LQ140),dtype=float)
mass_170 = 1700*np.ones(len(LQ170),dtype=float)
mass_200 = 2000*np.ones(len(LQ200),dtype=float)
mass_230 = 2300*np.ones(len(LQ230),dtype=float)


mass_arr = np.concatenate((mass_50,mass_80,mass_110,mass_140,mass_170,mass_200,mass_230),axis=0)
final_mass = np.c_[mass_arr,weight_arr]


frame = [LQ50,LQ80,LQ110,LQ140,LQ170,LQ200,LQ230]
LQ = pd.concat(frame)
LQ = LQ.dropna()


final_LQ = LQ.to_numpy(dtype=float)
finalTT = TT.to_numpy(dtype=float)
finalST1 = ST_1.to_numpy(dtype=float)
finalST2 = ST_2.to_numpy(dtype=float)
finalST3 = ST_3.to_numpy(dtype=float)
finalST4 = ST_4.to_numpy(dtype=float)
finalDY = DY.to_numpy(dtype=float)
finalWW = WW.to_numpy(dtype=float)
finalWZ = WZ.to_numpy(dtype=float)
finalZZ = ZZ.to_numpy(dtype=float)



# Scaling

scaler_mass = MinMaxScaler(feature_range=(0,1))
scaler_mass.fit(final_LQ)
final_LQ = scaler_mass.transform(final_LQ)
finalTT = scaler_mass.transform(finalTT)
finalST1 = scaler_mass.transform(finalST1)
finalST2 = scaler_mass.transform(finalST2)
finalST3 = scaler_mass.transform(finalST3)
finalST4 = scaler_mass.transform(finalST4)
finalDY = scaler_mass.transform(finalDY)
finalWW = scaler_mass.transform(finalWW)
finalWZ = scaler_mass.transform(finalWZ)
finalZZ = scaler_mass.transform(finalZZ)


#Model Definition
def mass_model():
    model = Sequential()
    model.add(Dense(len(lq_column), input_dim=len(lq_column)))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(200, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# Train-Test Split

data_train, data_test, mass_train, mass_test = train_test_split(final_LQ, mass_arr, test_size=0.1, random_state=32)
batch_size = 256




#Model Training
print("##############  TRAINING THE MASS MODEL OF REGRESSION TO RECONSTRUCT LQ MASS  ###############")

LQ_mass_model = mass_model()
es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=800)
mc = ModelCheckpoint('/work/yverma/DNN_SLQ/Regression_mass.h5', monitor='val_loss', mode='min', verbose=2, save_best_only=True)
history = LQ_mass_model.fit(data_train, mass_train, batch_size= batch_size, epochs=45, validation_data=(data_test,mass_test), shuffle=False,callbacks=[es, mc],verbose=2)

saved_model_mass = load_model('/work/yverma/DNN_SLQ/Regression_mass.h5')


#Loss and Accuracy history plots
print(history.history.keys())

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.savefig('/work/yverma/DNN_SLQ/Reg_loss.png')
plt.clf()
plt.close()



#Prediction of mass for all samples

tt_Predict = saved_model_mass.predict(finalTT)
st1_Predict = saved_model_mass.predict(finalST1)
st2_Predict = saved_model_mass.predict(finalST2)
st3_Predict = saved_model_mass.predict(finalST3)
st4_Predict = saved_model_mass.predict(finalST4)
dy_Predict = saved_model_mass.predict(finalDY)
ww_Predict = saved_model_mass.predict(finalWW)
wz_Predict = saved_model_mass.predict(finalWZ)
zz_Predict = saved_model_mass.predict(finalZZ)
lq_Predict = saved_model_mass.predict(final_LQ)



#Saving the predicted results into text files
LQ500_mass = []
LQ800_mass = []
LQ1100_mass = []
LQ1400_mass = []
LQ1700_mass = []
LQ2000_mass = []
LQ2300_mass = []
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
  if weight_arr[i] == sigma2:
    LQ500_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma3:
    LQ800_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma4:
    LQ1100_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma5:
    LQ1400_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma6:
    LQ1700_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma7:
    LQ2000_mass.append(lq_Predict[i][0])

for i in range(len(lq_Predict)):
  if weight_arr[i] == sigma8:
    LQ2300_mass.append(lq_Predict[i][0])





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




f1 = open("/work/yverma/DNN_SLQ/LQ500_mass.txt", "w+")

for i in LQ500_mass:
    f1.write(str(i))
    f1.write("\n")
f1.close()

f2 = open("/work/yverma/DNN_SLQ/LQ800_mass.txt", "w+")

for i in LQ800_mass:
    f2.write(str(i))
    f2.write("\n")
f2.close()

f3 = open("/work/yverma/DNN_SLQ/LQ1100_mass.txt", "w+")

for i in LQ1100_mass:
    f3.write(str(i))
    f3.write("\n")
f3.close()

f4 = open("/work/yverma/DNN_SLQ/LQ1400_mass.txt", "w+")

for i in LQ1400_mass:
    f4.write(str(i))
    f4.write("\n")
f4.close()

f5 = open("/work/yverma/DNN_SLQ/LQ1700_mass.txt", "w+")

for i in LQ1700_mass:
    f5.write(str(i))
    f5.write("\n")
f5.close()

f6 = open("/work/yverma/DNN_SLQ/LQ2000_mass.txt", "w+")

for i in LQ2000_mass:
    f6.write(str(i))
    f6.write("\n")
f6.close()


f7 = open("/work/yverma/DNN_SLQ/LQ2300_mass.txt", "w+")

for i in LQ2300_mass:
    f7.write(str(i))
    f7.write("\n")
f7.close()


f8 = open("/work/yverma/DNN_SLQ/TT_mass.txt", "w+")

for i in TT_mass:
    f8.write(str(i))
    f8.write("\n")
f8.close()


f9 = open("/work/yverma/DNN_SLQ/DY_mass.txt", "w+")

for i in DY_mass:
    f9.write(str(i))
    f9.write("\n")
f9.close()


f10 = open("/work/yverma/DNN_SLQ/ST1_mass.txt", "w+")

for i in ST1_mass:
    f10.write(str(i))
    f10.write("\n")
f10.close()


f11 = open("/work/yverma/DNN_SLQ/ST2_mass.txt", "w+")

for i in ST2_mass:
    f11.write(str(i))
    f11.write("\n")
f11.close()



f12= open("/work/yverma/DNN_SLQ/ST3_mass.txt", "w+")

for i in ST3_mass:
    f12.write(str(i))
    f12.write("\n")
f12.close()


f13 = open("/work/yverma/DNN_SLQ/ST4_mass.txt", "w+")

for i in ST4_mass:
    f13.write(str(i))
    f13.write("\n")
f13.close()


f14 = open("/work/yverma/DNN_SLQ/WW_mass.txt", "w+")

for i in WW_mass:
    f14.write(str(i))
    f14.write("\n")
f14.close()


f15 = open("/work/yverma/DNN_SLQ/WZ_mass.txt", "w+")

for i in WZ_mass:
    f15.write(str(i))
    f15.write("\n")
f15.close()

f16 = open("/work/yverma/DNN_SLQ/ZZ_mass.txt", "w+")

for i in ZZ_mass:
    f16.write(str(i))
    f16.write("\n")
f16.close()



