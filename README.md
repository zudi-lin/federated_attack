# Adver-Vis: Visualization of Adversarial Attacks

## Introduction

## Environment

The code is developed and tested under the following configurations.
- Hardware: 1-8 Nvidia GPUs (with at least 12G GPU memories) (change ```[--num-gpu GPUS]``` accordingly)
- Software: CentOS Linux 7.4 (Core), ***CUDA>=9.0, Python>=3.5, PyTorch>=1.0.0***

## Visulazation
* Visualize the training loss using [tensorboardX](https://github.com/lanpa/tensorboard-pytorch).
* Use TensorBoard with `tensorboard --logdir runs`  (needs to install TensorFlow).

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/zudi-lin/adver-vis/blob/master/LICENSE) file for details.