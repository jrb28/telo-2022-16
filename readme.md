# TELO-2022-16

This repository contains code that was used for a submission to the ACM Transactions of Evolutionary Learning and Optimization (TELO-2022-16).

# Computing Environments

The production runs of this code were run on a high-performance computing cluster, although the code in this repository is configured to run on a windows desktop environment.  Creating an equivalent computing environment on a Mac or Linux operating system may permit this code to be run, although we have not tried and and do not guarantee it.  Note that Macs have recently not been equipped with NVIDIA GPUs so that using a GPU to execute this code on a Mac is not feasible.  It is possible to run the code on a CPU although it is slower.  The code was run with Python 3.7.7 within an Anaconda environment that contained Tensorflow 1.14.  

- The Anaconda distribution can be downloaded from this site: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- Two environment files are located in this folder that can be used to create an Anaconda environment for this code.  One is for using the computer's CPU and the other the GPU if the computer is equipped with a cuda-enabled graphical processing unit.
  - `keras-cpu_tf14.yml`
  - `keras-gpu_tf14.yml`
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open an Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt with the environment of your choice:
    - <pre><code>conda env create -f <em>path_to_file</em>/keras-gpu_tf14.yml</code></pre>

# Executing the Code

## MNIST



## CIFAR-10
