# Hardware Implementation

This folder contains two artifacts:
- AutoBayes that converts a CNN to MCD-based BNN. 
- MCME FPGA-based accelerator, based on hls4ml and qkeras

## 1. Structure

```
.
├── README.md
├── autobayes            # All the scripts to generate data in the paper
    ├── models           # Keras models used for the experiment and testing
    ├── timing           # Timing reports generated here
    ├── diff_dropouts    # Synthesis reports for BNN with different number of dropout layers generated here
├── converter            # All code related to converting NN to BNN
    ├── keras            # Code for converting keras NN model to BNN model
    ├── pytorch          # Code for converting torch NN model to BNN model
├── mcme_hw              # FPGA-based hardware accelerator for MCME, based on qkears and hls4ml 
└── requirements.txt        
```

## 2. Limitation
Most of limitation comes from the unsupported features and instability of hls4ml and qkeras. We hope you are aware of before using it. 

### 2.1 Some known issues of dependencies
- Issue of qkeras on large model: Train is very instable. Sometimes different random seeds will make the training crash.
- Issue of hls4ml: The prediction of hls4ml on large model is weird. For example, the same scripts that work on Lenet will lead to accuracy problem in VGG and ResNet (the accuracy under qkeras has been valided. So the problem should be hls4ml). This is the reason why in the last part of experiments, we only evalute Bayes-LeNet.
- Automation between pytorch and qkeras: Some steps in our framework shown in Fig.3 of our paper are carried manually. For example, the selevtion of channel number of the co-design is performed in pytorch manually, and the optimization of hardware design parameters is also done by hand.

### 2.2 Future work
- Integrate lastest version of qkeras and hls4ml in our future version.
- More automatic integration between pytorch and qkeras part, and other optimization process of our framework.

## 3. Environment Setup

### 3.1 Dependencies Install

We can use conda to manage the environments.
```
conda create -n autobayes python=3.9
conda activate autobayes
pip3 install -r requirements.txt
```
If you are using Linux machine, make sure you set the environment variables for vivado
```
export LC_ALL=C; unset LANGUAGE
source /PATH/TO/Vivado/2019.2/settings64.sh
```

### 3.2 Dataset Download

Download `SVHN` dataset: 
```
cd ./mcme_hw/svhn
bash download_svhn.sh
```
The `**.mat` files will be downloaded under the same directory.

## 4. Artifact Evaluation

### 4.1 AutoBayes

This part converts a PyTorch or Keras neural network model to a Bayesian neural network model using Morte-Carlo Dropout.

All the scripts are put under the folder `./autobayes`, we refer reviewers/users to read [autobayes](./experiment/README.md) to run it.

### 4.2 MCME FPGA-based Accelerator

The synthesis and place&route may take several days to weeks depending on your machines. You can quickly check our reports in this [link](https://drive.google.com/drive/folders/1ldXGsGuJGxp8IPaYSD3CTrNEn71reTME?usp=sharing).

#### 4.2.1 Model Train and Prediction
The scripts are placed under `./mcme_hw/scripts/train_pred_eval/`. To run Lenet Experiment, follow:
```
cd mcme_hw
bash scripts/train_pred_eval/train_pred_mnist_lenet_mcme.sh
```
#### 4.2.1 Latency and Resource Evaluating
The scripts are placed under `./mcme_hw/scripts/lat_resource_eval/`. To run Lenet Experiment, follow:
```
cd mcme_hw
bash scripts/lat_resource_eval/mnist_lenet/cost_of_latency_lenet.sh  # Use Latency strategy to get results.
bash scripts/lat_resource_eval/mnist_lenet/cost_of_resource_lenet.sh  # Use Resource strategy to get results.
```
