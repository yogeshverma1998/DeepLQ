'''Author: Yogesh Verma'''
'''DeepLQ'''

#Calculating Significance from output of DeepLQ Binary Classification DNN and comparison with ST_MET

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

#weights
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
wwz = 0.00000690419
www = 0.00000950617
wdy = 0.00013864453
wzz = 0.00000608621


#definition of significance
def significance(lq,tt,dy,st1,st2,st3,st4,ww,wz,zz):
    sig_cut =[]
    for index in range(len(lq)):
	    sig = 150000*lq[index]*norm[index]/ np.sqrt(150000*(tt[index]*sigma1+ st1[index]*wst1 + dy[index]*wdy+ st2[index]*wst2 + st3[index]*wst3 + st4[index]*wst4 + ww[index]*www + wz[index]*wwz + zz[index]*wzz)) 
	    sig_cut.append(sig)
    return sig_cut



norm = [sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,sigma8]

'''Reading data from text files'''
with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/WW_st_met.txt") as ww:
    WW_st = [line.split() for line in ww]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/WZ_st_met.txt") as wz:
    WZ_st = [line.split() for line in wz]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/ZZ_st_met.txt") as zz:
    ZZ_st = [line.split() for line in zz]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/dy_st_met.txt") as dy:
    DY_st = [line.split() for line in dy]


with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/singletop1_st_met.txt") as st1:
    ST1_st = [line.split() for line in st1]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/singletop2_st_met.txt") as st2:
    ST2_st = [line.split() for line in st2]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/singletop3_st_met.txt") as st3:
    ST3_st = [line.split() for line in st3]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/singletop4_st_met.txt") as st4:
    ST4_st = [line.split() for line in st4]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/tt_st_met.txt") as tt:
    TT_st = [line.split() for line in tt]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ5_st_met.txt") as lq50:
    lQ50_st = [line.split() for line in lq50]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ8_st_met.txt") as lq80:
    lQ80_st = [line.split() for line in lq80]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ11_st_met.txt") as lq110:
    lQ110_st = [line.split() for line in lq110]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ14_st_met.txt") as lq140:
    lQ140_st = [line.split() for line in lq140]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ17_st_met.txt") as lq170:
    lQ170_st = [line.split() for line in lq170]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ20_st_met.txt") as lq200:
    lQ200_st = [line.split() for line in lq200]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Regression/LQ23_st_met.txt") as lq230:
    lQ230_st = [line.split() for line in lq230]



  

 

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/WW_binary_wet.txt") as ww:
    WW_wt = [line.split() for line in ww]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/WZ_binary_wet.txt") as wz:
    WZ_wt = [line.split() for line in wz]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/ZZ_binary_wet.txt") as zz:
    ZZ_wt = [line.split() for line in zz]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/DY_binary_wet.txt") as dy:
    DY_wt = [line.split() for line in dy]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/ST1_binary_wet.txt") as st1:
    ST1_wt = [line.split() for line in st1]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/ST2_binary_wet.txt") as st2:
    ST2_wt = [line.split() for line in st2]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/ST3_binary_wet.txt") as st3:
    ST3_wt = [line.split() for line in st3]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/ST4_binary_wet.txt") as st4:
    ST4_wt = [line.split() for line in st4]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/TT_binary_wet.txt") as tt:
    TT_wt = [line.split() for line in tt]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ500_binary_wet.txt") as lq50:
    lQ50_wt = [line.split() for line in lq50]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ800_binary_wet.txt") as lq80:
    lQ80_wt = [line.split() for line in lq80]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ1100_binary_wet.txt") as lq110:
    lQ110_wt = [line.split() for line in lq110]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ1400_binary_wet.txt") as lq140:
    lQ140_wt = [line.split() for line in lq140]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ1700_binary_wet.txt") as lq170:
    lQ170_wt = [line.split() for line in lq170]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ2000_binary_wet.txt") as lq200:
    lQ200_wt = [line.split() for line in lq200]

with open("/home/yogesh/PycharmProjects/UZH/Reco_Level/Binary_Classification/All_BKG/DNN_wet/LQ2300_binary_wet.txt") as lq230:
    lQ230_wt = [line.split() for line in lq230]


DY_arr_st = np.array(DY_st, dtype=float)
WW_arr_st = np.array(WW_st, dtype=float)
WZ_arr_st = np.array(WZ_st, dtype=float)
ZZ_arr_st = np.array(ZZ_st, dtype=float)
ST1_arr_st = np.array(ST1_st, dtype=float)
ST2_arr_st = np.array(ST2_st, dtype=float)
ST3_arr_st = np.array(ST3_st, dtype=float)
ST4_arr_st = np.array(ST4_st, dtype=float)
TT_arr_st = np.array(TT_st, dtype=float)
LQ50_arr_st = np.array(lQ50_st, dtype=float)
LQ80_arr_st = np.array(lQ80_st, dtype=float)
LQ110_arr_st = np.array(lQ110_st, dtype=float)
LQ140_arr_st = np.array(lQ140_st, dtype=float)
LQ170_arr_st = np.array(lQ170_st, dtype=float)
LQ200_arr_st = np.array(lQ200_st, dtype=float)
LQ230_arr_st = np.array(lQ230_st, dtype=float)


DY_arr_wt = np.array(DY_wt, dtype=float)
WW_arr_wt = np.array(WW_wt, dtype=float)
WZ_arr_wt = np.array(WZ_wt, dtype=float)
ZZ_arr_wt = np.array(ZZ_wt, dtype=float)
ST1_arr_wt = np.array(ST1_wt, dtype=float)
ST2_arr_wt = np.array(ST2_wt, dtype=float)
ST3_arr_wt = np.array(ST3_wt, dtype=float)
ST4_arr_wt = np.array(ST4_wt, dtype=float)
TT_arr_wt = np.array(TT_wt, dtype=float)
LQ50_arr_wt = np.array(lQ50_wt, dtype=float)
LQ80_arr_wt = np.array(lQ80_wt, dtype=float)
LQ110_arr_wt = np.array(lQ110_wt, dtype=float)
LQ140_arr_wt = np.array(lQ140_wt, dtype=float)
LQ170_arr_wt = np.array(lQ170_wt, dtype=float)
LQ200_arr_wt = np.array(lQ200_wt, dtype=float)
LQ230_arr_wt = np.array(lQ230_wt, dtype=float)






LQ_wt = [(LQ50_arr_wt,500),(LQ80_arr_wt,800),(LQ110_arr_wt,1100),(LQ140_arr_wt,1400),(LQ170_arr_wt,1700),(LQ200_arr_wt,2000),(LQ230_arr_wt,2300)]
LQ_st = [(LQ50_st,500),(LQ80_st,800),(LQ110_st,1100),(LQ140_st,1400),(LQ170_st,1700),(LQ200_st,2000),(LQ230_st,2300)]
TT_ST = [(TT_st,500),(TT_st,800),(TT_st,1100),(TT_st,1400),(TT_st,1700),(TT_st,2000),(TT_st,2300)]
ST1_ST = [(ST1_st,500),(ST1_st,800),(ST1_st,1100),(ST1_st,1400),(ST1_st,1700),(ST1_st,2000),(ST1_st,2300)]
ST2_ST = [(ST2_st,500),(ST2_st,800),(ST2_st,1100),(ST2_st,1400),(ST2_st,1700),(ST2_st,2000),(ST2_st,2300)]
ST3_ST = [(ST3_st,500),(ST3_st,800),(ST3_st,1100),(ST3_st,1400),(ST3_st,1700),(ST3_st,2000),(ST3_st,2300)]
ST4_ST = [(ST4_st,500),(ST4_st,800),(ST4_st,1100),(ST4_st,1400),(ST4_st,1700),(ST4_st,2000),(ST4_st,2300)]
WW_ST = [(WW_st,500),(WW_st,800),(WW_st,1100),(WW_st,1400),(WW_st,1700),(WW_st,2000),(WW_st,2300)]
WZ_ST = [(WZ_st,500),(WZ_st,800),(WZ_st,1100),(WZ_st,1400),(WZ_st,1700),(WZ_st,2000),(WZ_st,2300)]
ZZ_ST = [(ZZ_st,500),(ZZ_st,800),(ZZ_st,1100),(ZZ_st,1400),(ZZ_st,1700),(ZZ_st,2000),(ZZ_st,2300)]
DY_ST = [(DY_st,500),(DY_st,800),(DY_st,1100),(DY_st,1400),(DY_st,1700),(DY_st,2000),(DY_st,2300)]

TT_wt = [(TT_arr_wt,500),(TT_arr_wt,800),(TT_arr_wt,1100),(TT_arr_wt,1400),(TT_arr_wt,1700),(TT_arr_wt,2000),(TT_arr_wt,2300)]
ST1_wt = [(ST1_arr_wt,500),(ST1_arr_wt,800),(ST1_arr_wt,1100),(ST1_arr_wt,1400),(ST1_arr_wt,1700),(ST1_arr_wt,2000),(ST1_arr_wt,2300)]
ST2_wt = [(ST2_arr_wt,500),(ST2_arr_wt,800),(ST2_arr_wt,1100),(ST2_arr_wt,1400),(ST2_arr_wt,1700),(ST2_arr_wt,2000),(ST2_arr_wt,2300)]
ST3_wt = [(ST3_arr_wt,500),(ST3_arr_wt,800),(ST3_arr_wt,1100),(ST3_arr_wt,1400),(ST3_arr_wt,1700),(ST3_arr_wt,2000),(ST3_arr_wt,2300)]
ST4_wt = [(ST4_arr_wt,500),(ST4_arr_wt,800),(ST4_arr_wt,1100),(ST4_arr_wt,1400),(ST4_arr_wt,1700),(ST4_arr_wt,2000),(ST4_arr_wt,2300)]
WW_wt = [(WW_arr_wt,500),(WW_arr_wt,800),(WW_arr_wt,1100),(WW_arr_wt,1400),(WW_arr_wt,1700),(WW_arr_wt,2000),(WW_arr_wt,2300)]
WZ_wt = [(WZ_arr_wt,500),(WZ_arr_wt,800),(WZ_arr_wt,1100),(WZ_arr_wt,1400),(WZ_arr_wt,1700),(WZ_arr_wt,2000),(WZ_arr_wt,2300)]
ZZ_wt = [(ZZ_arr_wt,500),(ZZ_arr_wt,800),(ZZ_arr_wt,1100),(ZZ_arr_wt,1400),(ZZ_arr_wt,1700),(ZZ_arr_wt,2000),(ZZ_arr_wt,2300)]
DY_wt = [(DY_arr_wt,500),(DY_arr_wt,800),(DY_arr_wt,1100),(DY_arr_wt,1400),(DY_arr_wt,1700),(DY_arr_wt,2000),(DY_arr_wt,2300)]


final_tt = np.concatenate((TT_arr_wt,ST1_arr_wt,ST2_arr_wt,ST3_arr_wt,ST4_arr_wt),axis=0)






def number_generator(sample_list,prob_cut):
        lq_list = []
        for sample, mass in sample_list:
            #lq_list.append(len(sample[(sample>=mass-cut_width) & (sample<=mass+cut_width)]))
            num = len(sample[(sample>=prob_cut)])
            percent = float(num)/float(len(sample))
            #print(mass," DNN LQ ",percent)
            lq_list.append(num)
        return lq_list,prob_cut
        
def number_generator_st(sample_list,cut_width):
        lq_list = []
        ratio = []
        for sample, mass in sample_list:
            #lq_list.append(len(sample[(sample>=mass-cut_width) & (sample<=mass+cut_width)]))
            num = len(sample[(sample>=cut_width)])
            percent = float(num)/float(len(sample))
            ratio.append(percent)
            print(cut_width,mass,"ST_MET",percent)
            lq_list.append(num)
        return lq_list,cut_width,ratio



def tt_gen(ls,cut):
    tt_st =[]
    for index in range(len(ls)):
        sample = ls[index][0]
        mass = ls[index][1]
        num = len(sample[(sample>=cut)])
        tt_st.append(num)
    return tt_st
    
    
def tt_gen_st(ls,cut_width):
    tt_st =[]
    for index in range(len(ls)):
        sample = ls[index][0]
        mass = ls[index][1]
        num = len(sample[(sample>=cut_width)])
        tt_st.append(num)
    return tt_st


############ ST_MET ###################################
lq_st_2000,cut_2000,ratio = number_generator_st(LQ_st,2000)
tt_st_2000 = tt_gen_st(TT_ST,cut_2000)
st1_st_2000 = tt_gen_st(ST1_ST,cut_2000)
st2_st_2000 = tt_gen_st(ST2_ST,cut_2000)
st3_st_2000 = tt_gen_st(ST3_ST,cut_2000)
st4_st_2000 = tt_gen_st(ST4_ST,cut_2000)
ww_st_2000 = tt_gen_st(WW_ST,cut_2000)
wz_st_2000 = tt_gen_st(WZ_ST,cut_2000)
zz_st_2000 = tt_gen_st(ZZ_ST,cut_2000)
dy_st_2000 = tt_gen_st(DY_ST,cut_2000)
 
sig_cut_2000 = significance(lq_st_2000,tt_st_2000,dy_st_2000,st1_st_2000,st2_st_2000,st3_st_2000,st4_st_2000,ww_st_2000,wz_st_2000,zz_st_2000)
print("ST_MET")
print(lq_st_2000,tt_st_2000,dy_st_2000,st1_st_2000,st2_st_2000,st3_st_2000,st4_st_2000,ww_st_2000,wz_st_2000,zz_st_2000)
print(sig_cut_2000)
print(ratio)



############ DeepLQ Binary Classification ###################################
LQ_eff = [(LQ50_arr_wt,ratio[0]),(LQ80_arr_wt,ratio[1]),(LQ110_arr_wt,ratio[2]),(LQ140_arr_wt,ratio[3]),(LQ170_arr_wt,ratio[4]),(LQ200_arr_wt,ratio[5]),(LQ230_arr_wt,ratio[7])]


prob = np.linspace(0,1,10000)

def custom_limit(sample_list):
        lq_list = []
        cut_list = []
        for sample, eff in sample_list:
          for i in prob:
            #lq_list.append(len(sample[(sample>=mass-cut_width) & (sample<=mass+cut_width)]))
            num = len(sample[(sample>i)])
            percent = float(num)/float(len(sample))
            if percent >= eff: continue
            else:
            	print(eff," DNN LQ ",percent)
            	lq_list.append(num)
            	cut_list.append(i)
                break
        return lq_list,cut_list
    

lq_eff, cut_eff = custom_limit(LQ_eff)

def tt_custom(ls,cut_list):
    tt_st =[]
    for index in range(len(ls)):
        sample = ls[index][0]
        mass = ls[index][1]
        cut = cut_list[index]
        num = len(sample[(sample>cut)])
        tt_st.append(num)
    return tt_st


tt_eff = tt_custom(TT_wt,cut_eff)
st1_eff = tt_custom(ST1_wt,cut_eff)
st2_eff = tt_custom(ST2_wt,cut_eff)
st3_eff = tt_custom(ST3_wt,cut_eff)
st4_eff = tt_custom(ST4_wt,cut_eff)
ww_eff = tt_custom(WW_wt,cut_eff)
wz_eff = tt_custom(WZ_wt,cut_eff)
zz_eff = tt_custom(ZZ_wt,cut_eff)
dy_eff = tt_custom(DY_wt,cut_eff)
sig_eff = significance(lq_eff,tt_eff,dy_eff,st1_eff,st2_eff,st3_eff,st4_eff,ww_eff,wz_eff,zz_eff)
print("DeepLQ Binary Classification")
print(lq_eff,tt_eff,dy_eff,st1_eff,st2_eff,st3_eff,st4_eff,ww_eff,wz_eff,zz_eff)
print(sig_eff)




plt.plot(LQ_mass,sig_eff,'o--',label="DNN(Same efficiency)")
plt.plot(LQ_mass,sig_cut_2000,'v-',label="ST_MET >2000")
plt.title("Significance vs LQ_mass ")
plt.ylabel("Significance(log Scale)")
plt.xlabel("Mass[GeV]")
plt.legend(loc='best')
plt.yscale("log")
plt.savefig('Comp_eff.png')
plt.show()  




#% increase in Significance

def percent_increase(sig1,sig2):
    pinc = []
    for i in range(len(sig1)):
        if( sig1[i] > sig2[i] ):
           inc = 100*(sig1[i] - sig2[i])/sig2[i]
           pinc.append(inc)
        else:
           pinc.append(0)
    return pinc


pinc_dnn1_2 = percent_increase(sig_eff,sig_cut_2000)


plt.plot(LQ_mass,pinc_dnn1_2,'v--',label="DNN(Same Efficiency)")
plt.title("Percent Increase in Significance(DNN vs ST_MET(>2000)) ")
plt.ylabel("Percent Increase")
plt.xlabel("Mass[GeV]")
plt.legend(loc='best')
plt.savefig('Percent_inc.png')
plt.show()

















