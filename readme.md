# TELO-2022-16

This repo contains code that was used for a submission to the ACM Transactions of Evolutionary Learning and Optimization (TELO-2022-16).

# Computing Environments

The production runs of this code were run on a high-performance computing cluster, although the code in this repository is configured to run on a windows desktop environment.  Creating an equivalent computing environment on a Mac or Linux operating system may permit this code to be run, although we ahve not tried and and do not guarantee it.  The code was run with Python 3.8 within an Anaconda environment that contained Tensorflow 1.14.  

- The Anaconda distribution can be downloaded from this site: 
- Two environment files are located in this folder that can be used to create an Anaconda environment for this code.  One is for using the computer's CPU and the other the GPU if the computer is equipped with a cuda-enabled graphical processing unit.
  - keras-cpu_tf14.yml
  - keras-gpu_tf14.yml
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt
    - conda env create -f *path_to_file*/
