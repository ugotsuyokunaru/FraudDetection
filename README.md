# FraudDetection


## Topic Guide and Method Description
### TODO


## Usage of codes to reproduce our results
### 0. Repo Cloning
To start working on this project, clone this repository into your local machine by using the following command.
```
git clone https://github.com/ugotsuyokunaru/FraudDetection.git
```

### 1. Environment Building

(1) Python version required above 3.6.  
(2) Install stellargraph version 0.9.0b0
```
git clone https://github.com/stellargraph/stellargraph.git
cd stellargraph
pip install -r requirements.txt
pip install .
```
install reference :  
https://stellargraph.readthedocs.io/en/stable/quickstart.html?fbclid=IwAR1cyrmxdxRQnz4LJqCLioEwEzR0KDuS8T27AVbu3WxbkRtpHWM_-eBf0Oc#install-stellargraph-from-github-source

or another install way : 
```
pip install stellargraph
```

(3) Install other packages : 
```
pip install -r requirement.txt
```

### 2. Feature Engineering Preprocess
Use the following command to generate two pickle file  
1. ./data/combine_120days.pkl  
2. ./data/combine_30days.pkl  

for two different models with the origin train.zip and test.zip.
```
python preprocess.py
```

### 3. Training and Predicting
Use the following command to train three models and write prediction to csv files  
1. ./submit/focal.csv  
2. ./submit/david.csv  
3. ./submit/diff.csv  

before ensembling.  
```
python train.py
```

### 4. Ensembling
Use the following command to ensemble the above three models' prediction output files.
```
python ensemble.py
```