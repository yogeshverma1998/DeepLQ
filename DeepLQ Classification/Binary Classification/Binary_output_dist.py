'''Author: Yogesh Verma'''
'''DeepLQ'''
# ! /usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import ROOT
kBlue = ROOT.kBlue
kGreen = ROOT.kGreen
kRed = ROOT.kRed
kYellow = ROOT.kYellow
kBlack = ROOT.kBlack
kCyan = ROOT.kCyan
kGray = ROOT.kGray
kAzure = ROOT.kAzure
kMagenta = ROOT.kMagenta
kViolet = ROOT.kViolet
kSpring = ROOT.kSpring
kYellow  = ROOT.kYellow 
kPink  = ROOT.kPink
kTeal  = ROOT.kTeal
kMint = ROOT.kMint

#Weights used in scaling
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


#Reading files output from Binary Classification Model. Change the name of file w.r.t to model used
with open("/work/yverma/DNN_SLQ/ZZ_binary_wet.txt") as zz:
    ZZ_wt = [line.split() for line in zz]

with open("/work/yverma/DNN_SLQ/WW_binary_wet.txt") as ww:
    WW_wt = [line.split() for line in ww]

with open("/work/yverma/DNN_SLQ/WZ_binary_wet.txt") as wz:
    WZ_wt = [line.split() for line in wz]

with open("/work/yverma/DNN_SLQ/DY_binary_wet.txt") as dy:
    DY_wt = [line.split() for line in dy]

with open("/work/yverma/DNN_SLQ/ST1_binary_wet.txt") as st1:
    ST1_wt = [line.split() for line in st1]

with open("/work/yverma/DNN_SLQ/ST2_binary_wet.txt") as st2:
    ST2_wt = [line.split() for line in st2]

with open("/work/yverma/DNN_SLQ/ST3_binary_wet.txt") as st3:
    ST3_wt = [line.split() for line in st3]

with open("/work/yverma/DNN_SLQ/ST4_binary_wet.txt") as st4:
    ST4_wt = [line.split() for line in st4]

with open("/work/yverma/DNN_SLQ/TT_binary_wet.txt") as tt:
    TT_wt = [line.split() for line in tt]

with open("/work/yverma/DNN_SLQ/LQ500_binary_wet.txt") as lq50:
    lQ50_wt = [line.split() for line in lq50]

with open("/work/yverma/DNN_SLQ/LQ800_binary_wet.txt") as lq80:
    lQ80_wt = [line.split() for line in lq80]

with open("/work/yverma/DNN_SLQ/LQ1100_binary_wet.txt") as lq110:
    lQ110_wt = [line.split() for line in lq110]

with open("/work/yverma/DNN_SLQ/LQ1400_binary_wet.txt") as lq140:
    lQ140_wt = [line.split() for line in lq140]

with open("/work/yverma/DNN_SLQ/LQ1700_binary_wet.txt") as lq170:
    lQ170_wt = [line.split() for line in lq170]

with open("/work/yverma/DNN_SLQ/LQ2000_binary_wet.txt") as lq200:
    lQ200_wt = [line.split() for line in lq200]

with open("/work/yverma/DNN_SLQ/LQ2300_binary_wet.txt") as lq230:
    lQ230_wt = [line.split() for line in lq230]




ZZ_arr_wt = np.array(ZZ_wt, dtype=float)
WW_arr_wt = np.array(WW_wt, dtype=float)
WZ_arr_wt = np.array(WZ_wt, dtype=float)
DY_arr_wt = np.array(DY_wt, dtype=float)
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


#Plotting output distributions
c1 = ROOT.TCanvas('c1','Example',2700,1900)
legend = ROOT.TLegend(0.5,0.6,0.88,0.88);
hist = ROOT.TH1F("plot1","",100,0,1)
for i in DY_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist.Fill(i,wdy)
    hist.SetMinimum(1E-11)
    hist.SetMaximum(1E+4)
    #hist.SetMarkerStyle(ROOT.kFullCircle)
    hist.SetLineColor(kCyan)
    hist.SetLineWidth(4)
    hist.SetTitle("")
    hist.SetXTitle("Probability")
    hist.SetYTitle("#sigma / N_{events}^{gen}")


c1.SetLogy()
hist.Draw()

hist0 = ROOT.TH1F("plot0","",100,0,1)
for i in TT_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist0.Fill(i,sigma1)
    hist0.SetMinimum(1E-11)
    hist0.SetMaximum(1E+4)
    #hist0.SetMarkerStyle(ROOT.kFullCircle)
    hist0.SetLineColor(kViolet)
    hist0.SetLineWidth(4)
    hist0.SetTitle("")
    hist.SetXTitle("Probability")
    hist0.SetYTitle("#sigma / N_{events}^{gen}")


hist0.Draw("Same")


hist2 = ROOT.TH1F("plot2","",100,0,1)
for i in LQ50_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist2.Fill(i,sigma2)
    hist2.SetMinimum(1E-11)
    hist2.SetMaximum(1E+4)
    hist2.SetMarkerStyle(ROOT.kFullCircle)
    hist2.SetMarkerColor(kGreen)
    hist2.SetTitle("")
    hist2.SetXTitle("Probability")
    hist2.SetYTitle("#sigma / N_{events}^{gen}")


hist2.Draw("Same")

hist3 = ROOT.TH1F("plot3","",100,0,1)
for i in LQ80_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist3.Fill(i,sigma3)
    hist3.SetMinimum(1E-11)
    hist3.SetMaximum(1E+4)
    hist3.SetMarkerStyle(ROOT.kFullCircle)
    hist3.SetMarkerColor(kRed)
    hist3.SetTitle("")
    hist3.SetXTitle("Probability")
    hist3.SetYTitle("#sigma / N_{events}^{gen}")


hist3.Draw("Same")


hist4 = ROOT.TH1F("plot4","",100,0,1)
for i in LQ110_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist4.Fill(i,sigma4)
    hist4.SetMinimum(1E-11)
    hist4.SetMaximum(1E+4)
    hist4.SetMarkerStyle(ROOT.kFullCircle)
    hist4.SetMarkerColor(kYellow)
    hist4.SetTitle("")
    hist4.SetXTitle("Probability")
    hist4.SetYTitle("#sigma / N_{events}^{gen}")


hist4.Draw("Same")



hist5 = ROOT.TH1F("plot5","",100,0,1)
for i in LQ140_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist5.Fill(i,sigma5)
    hist5.SetMinimum(1E-11)
    hist5.SetMaximum(1E+4)
    hist5.SetMarkerStyle(ROOT.kFullCircle)
    hist5.SetMarkerColor(kBlack)
    hist5.SetTitle("")
    hist5.SetXTitle("Probability")
    hist5.SetYTitle("#sigma / N_{events}^{gen}")


hist5.Draw("Same")


hist6 = ROOT.TH1F("plot6","",100,0,1)
for i in LQ170_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist6.Fill(i,sigma6)
    hist6.SetMinimum(1E-11)
    hist6.SetMaximum(1E+4)
    hist6.SetMarkerStyle(ROOT.kFullCircle)
    hist6.SetMarkerColor(kAzure)
    hist6.SetTitle("")
    hist6.SetXTitle("Probability")
    hist6.SetYTitle("#sigma / N_{events}^{gen}")



hist6.Draw("Same")



hist7 = ROOT.TH1F("plot7","",100,0,1)
for i in LQ200_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist7.Fill(i,sigma7)
    hist7.SetMinimum(1E-11)
    hist7.SetMaximum(1E+4)
    hist7.SetMarkerStyle(ROOT.kFullCircle)
    hist7.SetMarkerColor(kGray)
    hist7.SetTitle("")
    hist7.SetXTitle("Probability")
    hist7.SetYTitle("#sigma / N_{events}^{gen}")



hist7.Draw("Same")


hist8 = ROOT.TH1F("plot8","",100,0,1)
for i in LQ230_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist8.Fill(i,sigma8)
    hist8.SetMinimum(1E-11)
    hist8.SetMaximum(1E+4)
    hist8.SetMarkerStyle(ROOT.kFullCircle)
    hist8.SetMarkerColor(kMagenta)
    hist8.SetTitle("")
    hist8.SetXTitle("Probability")
    hist8.SetYTitle("#sigma / N_{events}^{gen}")



hist8.Draw("Same")


hist_st1 = ROOT.TH1F("plotst1","",100,0,1)
for i in ST1_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_st1.Fill(i)
    hist_st1.SetMinimum(1E-11)
    hist_st1.SetMaximum(1E+4)
    hist_st1.SetMarkerStyle(ROOT.kFullCircle)
    hist_st1.SetMarkerColor(kGray)
    hist_st1.SetTitle("")
    hist_st1.SetXTitle("Probability")
    hist_st1.SetYTitle("#sigma / N_{events}^{gen}")




hist_st2 = ROOT.TH1F("plotst2","",100,0,1)
for i in ST2_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_st2.Fill(i)
    hist_st2.SetMinimum(1E-11)
    hist_st2.SetMaximum(1E+4)
    hist_st2.SetMarkerStyle(ROOT.kFullCircle)
    hist_st2.SetMarkerColor(kGray)
    hist_st2.SetTitle("")
    hist_st2.SetXTitle("Probability")
    hist_st2.SetYTitle("#sigma / N_{events}^{gen}")



hist_st3 = ROOT.TH1F("plotst3","",100,0,1)
for i in ST3_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_st3.Fill(i)
    hist_st3.SetMinimum(1E-11)
    hist_st3.SetMaximum(1E+4)
    hist_st3.SetMarkerStyle(ROOT.kFullCircle)
    hist_st3.SetMarkerColor(kGray)
    hist_st3.SetTitle("")
    hist_st3.SetXTitle("Probability")
    hist_st3.SetYTitle("#sigma / N_{events}^{gen}")



hist_st4 = ROOT.TH1F("plotst4","",100,0,1)
for i in ST4_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_st4.Fill(i)
    hist_st4.SetMinimum(1E-11)
    hist_st4.SetMaximum(1E+4)
    hist_st4.SetMarkerStyle(ROOT.kFullCircle)
    hist_st4.SetMarkerColor(kGray)
    hist_st4.SetTitle("")
    hist_st4.SetXTitle("Probability")
    hist_st4.SetYTitle("#sigma / N_{events}^{gen}")



hist_ww = ROOT.TH1F("plotww","",100,0,1)
for i in WW_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_ww.Fill(i)
    hist_ww.SetMinimum(1E-11)
    hist_ww.SetMaximum(1E+4)
    hist_ww.SetMarkerStyle(ROOT.kFullCircle)
    hist_ww.SetMarkerColor(kGray)
    hist_ww.SetTitle("")
    hist_ww.SetXTitle("Probability")
    hist_ww.SetYTitle("#sigma / N_{events}^{gen}")



hist_zz = ROOT.TH1F("plotzz","",100,0,1)
for i in ZZ_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_zz.Fill(i)
    hist_zz.SetMinimum(1E-11)
    hist_zz.SetMaximum(1E+4)
    hist_zz.SetMarkerStyle(ROOT.kFullCircle)
    hist_zz.SetMarkerColor(kGray)
    hist_zz.SetTitle("")
    hist_zz.SetXTitle("Probability")
    hist_zz.SetYTitle("#sigma / N_{events}^{gen}")


hist_wz = ROOT.TH1F("plotwz","",100,0,1)
for i in WZ_arr_wt:
    ROOT.gStyle.SetOptStat(0)
    hist_wz.Fill(i)
    hist_wz.SetMinimum(1E-11)
    hist_wz.SetMaximum(1E+4)
    hist_wz.SetMarkerStyle(ROOT.kFullCircle)
    hist_wz.SetMarkerColor(kGray)
    hist_wz.SetTitle("")
    hist_wz.SetXTitle("Probability")
    hist_wz.SetYTitle("#sigma / N_{events}^{gen}")


hist_fn = ROOT.TH1F("plotfn","",100,0,1)
hist_fn.Add(hist_st1,wst1)
hist_fn.Add(hist_st2,wst2)
hist_fn.Add(hist_st3,wst3)
hist_fn.Add(hist_st4,wst4)
hist_fn.SetLineColor(kBlue)
hist_fn.SetLineWidth(4)
hist_fn.Draw("Same")
hist_db = ROOT.TH1F("plotdb","",100,0,1)
hist_db.Add(hist_ww,www)
hist_db.Add(hist_wz,wwz)
hist_db.Add(hist_zz,wzz)
hist_db.SetLineColor(kTeal)
hist_db.SetLineWidth(4)
hist_db.Draw("Same")
legend.SetNColumns(2)
legend.SetTextSize(0.02)
legend.AddEntry(hist_fn,"ST","f")
legend.AddEntry(hist_db,"Diboson","f")
legend.AddEntry(hist,"DY","f")
legend.AddEntry(hist0,"TTbar","f")
legend.AddEntry(hist2,"LQ(m = 500)","lep")
legend.AddEntry(hist3,"LQ(m = 800)","lep")
legend.AddEntry(hist4,"LQ(m = 1100)","lep")
legend.AddEntry(hist5,"LQ(m = 1400)","lep")
legend.AddEntry(hist6,"LQ(m = 1700)","lep")
legend.AddEntry(hist7,"LQ(m = 2000)","lep")
legend.AddEntry(hist8,"LQ(m = 2300)","lep")
legend.SetBorderSize(0)
legend.Draw()
c1.Print("/work/yverma/DNN_SLQ/Binary_reconst.png")
