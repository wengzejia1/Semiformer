# Semiformer 

We introduce a joint semi-supervised learning framework, Semiformer, which contains a transformer stream, a convolutional stream and a carefully designed fusion module for knowledge sharing between these streams. The convolutional stream is trained on limited labeled data and further used to generate pseudo labels to supervise the training of the transformer stream on unlabeled data.

# Getting started

PyTorch 1.7.0+ and torchvision 0.8.1+ and [pytorch-image-models 0.3.2](https://github.com/rwightman/pytorch-image-models)

# Log and Checkpoint downloaded

You can download the log file and the checkpoint file from the following links:

- log: https://drive.google.com/file/d/1oR2e1AP-luOGPPoKWiR03sYE7kTMWdqZ/view?usp=sharing
- checkpoint: https://drive.google.com/file/d/1iWqgVMea9hlU-lBEMsGT-sptFR-PwPr2/view?usp=sharing

# Data Preparation

The directory structure of ImageNet is expected as:

```
/path/to/imagenet
  train/
    c1/
    c2/
    ...
    c1000/
  val/
    c1/
    c2/
    ...
    c1000/
```

# Evaluation

You can download the checkpoint file and evaluate the model by the script "eval.sh" in the /script folder. You will get the 75.5 Top1 Accuracy.

```
# set $ROOT as the project root path
# set $DATA_ROOT as the INet path 
# SET $RESUME_PATH as the downloaded checkpoint path
bash eval.sh
```

# Train

Training scripts of Semiformer are provided in /script/submitit_Semiformer_*.sh using submitit. You can also train the model in a simple ddp mannor, referring to the DDP script example provided in /script/run_ddp_example.sh. 
