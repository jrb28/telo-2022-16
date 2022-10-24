# TELO-2022-16

This repository contains code that was used for a submission to the ACM Transactions of Evolutionary Learning and Optimization (TELO-2022-16) for generating adversarial examples for the MNIST[^mnist] and CIFAR-10[^cifar] (henceforth, CIFAR) data sets.
- [MNIST](https://github.com/telo-author/telo-2022-16/tree/main/MNIST)
- [CIFAR](https://github.com/telo-author/telo-2022-16/tree/main/CIFAR)

# Computing Environments

Each of the MNIST and CIFAR folders have a `readme.md` file that gives instructions for creating an appropriate Anaconda environment for executing the code in these folders.

The production runs of this code were run on a high-performance computing cluster, although the code in this repository is configured to run on a Windows desktop operating system.  It may be possible to successfully run this code on a Mac or a Linux operating system by creating an equivalent computing environment as described in the MNIST and CIFAR subfolders but we have not tried to do so and we do not guarantee success on those operating systems.  In addition, note that Macs have not recently been equipped with NVIDIA GPUs so that using a GPU to execute this code on a Mac is not feasible.  Code is also provided to run on a CPU, with multiprocessing, if desired.    





[^mnist]: See xxx and xxx
[^cifar]: See xxx
