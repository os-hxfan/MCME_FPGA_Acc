# Software Implementation
This github repo contains part of the codebase for the DAC 2023 conference paper "When Monte-Carlo Dropout Meets Multi-Exit:
Optimizing Bayesian Neural Networks on FPGA". The paper will be available on ArXiv shortly.

## 1. Structure 
    ├── README.md    
    ├── requirements.txt
    ├── script_tables          # Scripts to generate the tables in the paper
    ├── software               # Code to build and train multi-exit BNNs
        ├── main.py
        ├── datasets           # Code containing dataset files and loaders
           ├── data            # Code containing the train/test data
        ├── models             # Code containing model files and loaders
           ├── model_weights   # Code containing model weight files
        ├── train              # Code containing training files
           ├── loss            # Code containing loss loaders
        ├── snapshots          # Checkpoints of the model during training
           ├── figures         # Figures of the loss vs epoch
        ├── runs_db            # Database containing hyperparameters and performance of each experiment
        ├── logs               # Logs containing outputs used in paper
           
## 2. Environment Setup

### 2.1 Install Dependencies
```
pip install https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchaudio-0.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install https://download.pytorch.org/whl/cu113/torchvision-0.12.0%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install -r requirements.txt
```

### 2.2 Prepare Datasets

The data folder should be made to look like the below:

    ├── chestx
       ├── Data_Entry_2017.csv
       ├── images
    ├── cifar100
       ├── cifar-100-batches-py
         
    
The datasets can be downloaded here:
ChestX-ray 14: https://www.kaggle.com/nih-chest-xrays/data,
Cifar100: https://pytorch.org/vision/stable/datasets.html,

To get the ChestX-ray 14 dataset in the correct format, each of the image folders which are unpacked from the download (images_001, images_002,...) need to be combined into a single folder "images".

## 2.3 How to Run

Before executing the script in script_figs, run the following commands

```
mkdir software/snapshots
mkdir software/logs
mkdir software/runs_db
mkdir software/snapshots/figures
```

Following that, simply run the script in the script_figs directory to reproduce the results used in Table 1 of the paper. We note that the standard errors were calculated through averaging the results of 5 trials for each individual hyperparameter setting. All results will be saved in a log file with the following order:
Model Type, Accuracy, ECE, FLOPs as a fraction of the baseline

### 2.3 Model Type: 
The Model Type column has the following format:

E (*dropout_rate*, *Exit* or *Ensemble Number* or *Confidence Threshold*)

Where *Exit* (i.e. 0, 1, 2...) refers to the performance of the predictions from only that specific exit, with exit 0 being the earliest possible exit, *Ensemble Number* (i.e. Ensemble0, Ensemble1...) refers to the performance of the ensembled predictions from all previous exits (i.e. Ensemble1 averages the predictions of Exit 0 and Exit 1), and *Confidence Threshold* (i.e. 0.1, 0.15...) refers to the performance of the predictions utilizing early exiting with that specific confidence threshold.

For the Baseline and standard Monte-Carlo networks, only the final exit result has meaning (i.e. E (*dropout_rate*, 3) for ResNet18 or E (*dropout_rate*, 4) for VGG19)

## 3. Limitation and Future Work

### 3.1 Limitation
- The selection of Exit and MCD parameters are not automatic
- Improvement is not very impressive
- Evaluation of uncertainty metrics 

### 3.2 Future work 
- Better metrics for 