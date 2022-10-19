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

Adversarial examples can be generated in two modes: 

- A "worker" program can be executed from the command line to generate one adversarial for a specified MNIST image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times for a sequence of MNIST images.  The controller program uses multiprocessing.

### Using the Worker Program

Two worker programs exist, one for Feedforward Neural Networks  (FFs) and a separate program for Convolutional Neural Networks (CNNs):
- FF: `ga_mnist_adv_worker_rank-sel.py`
- CNN: `ga_mnist_adv_worker_rank-sel_cnn.py`
The programs can be run from the command line or by specifying command line arguments in an IDE, for example, in Anaconda Spyder by choosing ``Run>Configuration per file`` and then specifying input arguments in the ``Command line options`` dialog field.

The input arguments for these command line programs are as follows, in this order:
- `mnist_id`: an integer from 0 to 59,999
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin, mad-recip, mad-linear`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `file_model`: either `FF.json`` or `CNN.json``
- `file_weights`: either `FF.h5` or CNN.h5`
- `out_folder`: 
- `in_folder`:
- `pop_size`: an integer representing the population size
- `prob_mut_genome`:
- `prob_mut_pixel`:
- `num_gen`: an integer representing the number of generations to be executed
- `scen_name`: A scenario name to be associated with this set of parameters.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_mnist_adv_worker_rank-sel.py 0 L2 rank-linear rand 0.9 FF.json FF.h5 xxx xxx 1000 1.0 xxx  2000 0_L2_rl_rand_FF_1000_2000<\code></pre>
A similar command can be used with `ga_mnist_adv_worker_rank-sel_cnn.py` although the neural network files, `CNN.json` and `CNN.h5``, wuld be referenced in the input arguments.

Output from this code includes the follosing items:
- xxxx


### Using the Controller Program

The controller program executes multiple worker files in parallel.  Its input arguments are:
- `start_id`: an integer from 0 to 59,999 representing the first MNIST iamge for which an adversarial example is generated
- `end_id`: an integer from 0 to 59,999 representing the last MNIST iamge for which an adversarial example is generated
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin, mad-recip, mad-linear`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `num_proc`: number of processors to be used in parallel mode
- `file_model`: either `FF.json`` or `CNN.json``
- `file_weights`: either `FF.h5` or CNN.h5`
- `folder_in`: 
- `folder_out`:
- `filepath_code`: the filepath (and filename) for the pyrrhon executable worker file `ga_mnist_adv_worker_rank-sel.py` or `ga_mnist_adv_worker_rank-sel_cnn.py`
- `pop_size`: an integer representing the population size
- `prob_mut_genome`:
- `pixel_mut_per_phenotype`:
- `num_gen`: an integer representing the number of generations to be executed




## CIFAR-10
