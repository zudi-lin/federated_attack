# Federated Adversarial Attacks in ConvNets

## Introduction

Abundant of previous works have indicated that convolutional networks (ConvNets) are vulnerable to **adversarial attacks**, which only introduce quasi-imperceptible perturbations to the input images but can completely deceive the models. However, such attacks have not been fully investigated under the federated scheme, where a substantial amount of local devices work collaboratively to create a robust model. In this project, we investigate the adversarial attack under the **federated learning** setting so that the center server aggregates the adversarial examples of local devices without access to users' data. We show that the aggregated attack examples are more robust and can deceive more models with different architecture and training data. We also implement a system with cloud storage to efficiently simulate the environment and facilitate future researches.

## Notes

This repository constains the codes of the project of course CS244R at Harvard University. We thank Prof. HT Kung, Marcus  Comiter and Sai Qian Zhang for the help and guidance on this project.

## Environment

The code is developed and tested under the following configurations.
- Hardware: 1-8 Nvidia GPUs (with at least 12G GPU memories) (change ```[--num-gpu GPUS]``` accordingly)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=9.0, Python>=3.6, PyTorch>=1.0.0***

## Installation
```
conda create -n mypython3 python=3.6
source activate activate py3_torch
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
pip install -r requirements.txt
```

## Running

### Federated Attack System: 
- First, run ```python system_script/[real_system_with_adaptive_buffer_read]global_server.py```
- Second, run ```python serveral job of system_script/[real_system_with_adaptive_buffer_read]local_node.py```

### Imbalanced Upload Solution Simulation:
- First, run ```python system_script/[imbalanced_upload_simulation]global_server.py```
- Second, run ```python serveral job of system_script/[imbalanced_upload_simulation]local_node.py```

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/adver-vis/blob/master/LICENSE) file for details.

