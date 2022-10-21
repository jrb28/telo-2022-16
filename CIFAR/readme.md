This folder contains code for generating adversarial examples for the CIFAR-10 data set.

# Computing Environments

The production runs of this code were run on a high-performance computing cluster, although the code in this repository is configured to run on a windows desktop environment.  Creating an equivalent computing environment on a Mac or Linux operating system may permit this code to be run, although we have not tried and and do not guarantee it.  Note that Macs have recently not been equipped with NVIDIA GPUs so that using a GPU to execute this code on a Mac is not feasible.  It is possible to run the code on a CPU although it is slower.  The code was run with Python 3.8.8 within an Anaconda environment that contained Tensorflow 2.7.  

- The Anaconda distribution can be downloaded from this site: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- An environment file is located in this folder that can be used to create an Anaconda environment for this code.  
  - `tf2.yml`
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open an Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt with the environment of your choice:
    - <pre><code>conda env create -f <em>path_to_file</em>/tf2.yml</code></pre>

For a GPU to be successfully used, it be `cuda` enabled and, furthermore, graphics drivers must be up to date with the appropriate versions of `cuda` and `cudnn`  installed.   Success with a GPU is also dependent, of course, on the GPU having sufficient memory.

# Executing the Code

Adversarial examples can be generated in two modes: 
- A "worker" program can be executed from the command line to generate one adversarial for a specified CIFAR-10 image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times for a sequence of MNIST images.  The controller program uses multiprocessing.

The `code`, `input`, and `output` folders are located in this root folder for the `CIFAR` coder.  These code files can be executed on either with a CPU or a GPU by supplying an appropriate input argument as described below.

## Using the Worker Program

The filename for the worker program is `ga_cifar_worker.py`.  It can be run from the command line or by specifying command line arguments in an IDE, for example, in Anaconda Spyder by choosing ``Run>Configuration per file`` and then specifying input arguments in the ``Command line options`` dialog field.

The input arguments for these command line programs are as follows, in this order:
- `cifar_id`: an integer from 0 to 59,999
- `model_file`: `model2.h5`
- `out_folder`: 
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: `bright` for brightness mutation
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `gpu_mode`: either `FF.json` or `CNN.json`

There are fewer input arguments for the CIFAR-10 code compared with the MNIST code because the parameters that were optimized during the initial test runs and that were invariant in the experiments were hard-coded indto the program file.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_cifar_worker.py 0 xxx xxx L2 rank-linear rand 0.9 xxx<\code></pre>

If `gpu_mode` is `False`, then `tensorflow 2.0` will automatically use multiprocessing on however many CPUU cores are available.


Output from this code includes the following items:
- xxxx


## Using the Controller Program

The controller program filename is `ga_cifar_controlr.py` and it executes multiple worker files in parallel.  Its input arguments are:
- `start_id`: an integer from 0 to 59,999 representing the first CIFAR-10 iamge for which an adversarial example is generated
- `end_id`: an integer from 0 to 59,999 representing the last CIFAR-10 iamge for which an adversarial example is generated
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin, mad-recip, mad-linear`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `num_proc`: number of processors to be used in parallel mode
- `file_model`: `model2.h5`
- `folder`: 
- `gpu_mode`:
- `mp_mode`: the filepath (and filename) for the pyrrhon executable worker file `ga_mnist_adv_worker_rank-sel.py` or `ga_mnist_adv_worker_rank-sel_cnn.py`
- `batch_id`: an integer representing the population size

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_cifar_control.py 0 19 L2 rank-linear rand 0.9 10 covnet.h5 xxx xxx xxx xxx</code></pre>

Output from this code includes the following items:
- xxxx

