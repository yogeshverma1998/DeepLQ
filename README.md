# DeepLQ
DNN-based novel reconstruction and classification approaches in the search for LQs → bτ. Work is submitted as a CMS Experiment Internal Analysis Note-2020. 

# Data
Data is generated from official CMSSW 10.2.10  Monte Carlo (MC) Simulator with the data sample (2016) of pp collisions at a √s = 13 TeV


# Required Libraries
```js
python 3.6.8
ROOT + PyROOT 6.2
Tensorflow 1.1.4
Keras > 2.0
```
All other standard libraries like pandas, numpy etc are required. Detailed list and installation in server using anaconda environment is provided here: https://github.com/kschweiger/TF4ttHFH/blob/master/setup_t3PSI.sh

# Training
To train the DeepLQ model, choose the type of model from Regression or Classification and execute following. Remenber to change the input data files in each Model_filename.py to actual location of your dataset
```js
python3 Model_filename.py
```
