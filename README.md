# FraudDetection

## Usage of codes to reproduce our results

### 0. Repo Cloning
To start working on this project, clone this repository into your local machine by using the following command.

    git clone https://github.com/ugotsuyokunaru/FraudDetection.git

### 1. Environment Building
Python version required above 3.6.

    pip install -r requirement.txt

### 2. Feature Engineering Preprocess
Use the following command to generate two pickle file (for two different models) with the origin train.zip and test.zip.

    python ./src/preprocess.py

### 3. Training

    python ./src/train.py