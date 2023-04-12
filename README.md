# MCME_FPGA_Acc
This repo contains the artifacts of our DAC'23 paper:
`When Monte-Carlo Dropout Meets Multi-Exit: Optimizing Bayesian Neural Networks on FPGA`. 
The software implementation is based on `Pytorch`, and the hardware implementation is based on `HLS4ML` and `QKeras`. Pls cite our paper and give this repo :star: if you feel the code and paper are helpful!

## Structure
```
.
├── README.md  
├── Software_Artifact                     # Evaluating accuracy and ECE for MCME-based networks
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
├── Hardware_Artifact          # Evaluating hardware performance of proposed FPGA-based accelerator
  ├── README.md
  ├── autobayes                # All the scripts to generate data in the paper
      ├── models               # Keras models used for the experiment and testing
      ├── timing               # Timing reports generated here
      ├── diff_dropouts        # Synthesis reports for BNN with different number of dropout layers generated here
  ├── converter                # All code related to converting NN to BNN
      ├── keras                # Code for converting keras NN model to BNN model
      ├── pytorch              # Code for converting torch NN model to BNN model
  ├── mcme_hw                  # FPGA-based hardware accelerator for MCME, based on qkears and hls4ml 
  └── requirements.txt   
```

## How to Run

1. Setup environment by following `README.md` under both `./Software_Artifact` and `./Hardware_Artifact`.  Virtual environment management system such as [Conda](https://docs.conda.io/en/latest/) is highly recommended.
2. Run the scripts under both folder to generate the results shown in the paper.

## Limitation and Future Work

Pls read the limitation sections under both `./Software_Artifact` and `./Hardware_Artifact` folders.

## Acknowledgement

We thanks out MSc students, Liam (castelliliam@gmail.com) and Mark (hao.chen20@imperial.ac.uk), in contributing the code during their MSc project and summer intern.