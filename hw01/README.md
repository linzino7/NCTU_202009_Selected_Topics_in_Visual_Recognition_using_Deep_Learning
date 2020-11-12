# HW1: Image Classification


## Hardware
- Ubuntu 18.04 LTS
- Intel(R) Xeon(R) CPU E5-2696 v4 @ 2.20GHz
- 62.9G RAM
- NVIDIA T4

## Reproducing Submission
To reproduct my submission without retrainig, do the following steps:
1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Training&Inference](#Training&Inference)
4. [Submission](#submission)

## Installation
All requirements should be detailed in requirements.txt. Using Anaconda is strongly recommended.
```
pip install -r requirements.txt
```

## Dataset Preparation
All required files except images are already in data directory.
If you generate CSV files (duplicate image list, split, leak.. ), original files are overwritten. The contents will be changed, but It's not a problem.

### Prepare Images
After downloading and converting images, the data directory is structured as:
```
data
  +- training_data
  |  +- training_data
  +- testing_data
  |  +- testing_data
  +- training_labels.csv
```

#### Download Official Image
Download and extract *cs-t0828-2020-hw1.zi* to *dataw* directory.
If the Kaggle API is installed, run following command.
```
$ kaggle competitions download -c cs-t0828-2020-hw1
$ mkdir data
$ unzip cs-t0828-2020-hw1.zip -d data
```


##  Train models
To train models, run following commands.
```
$ python hw01.py 
```
The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet50 | 1x NVIDIA T4 | 224*224 | 100 | 5 hours


## Submission
Following command will ensemble of all models and make submissions.
```
$ kaggle competitions submit -c cs-t0828-2020-hw1 -f model_hw01_restnet50_100_16_0.005_sgd_all_nor_slr.csv -m "Message"
```
