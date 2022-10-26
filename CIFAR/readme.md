# CIFAR-10 Adversarial Examples

This folder contains code for generating adversarial examples for the CIFAR-10 data set.

## Subfolders in this Repository

- `code`: code for generating adversarial examples
- `input`: input files required to run the code
- `output`: folder for receiving output
- `adv_egs`: a static folder that contains the adversarial examples that were reported in in the article.  A `numpy` file is included in this folder for every parameter set that was evaluated with 500 adversarial examples for the MNIST IDs in `../input/randMNIST.json`.  The filename indicates the genetic algorithm parameters joined by underscores. 

## Computing Environment

The production runs of this code were run on a high-performance computing cluster, although the code in this repository is configured to run on a windows desktop environment.  Creating an equivalent computing environment on a Mac or Linux operating system may permit this code to be run, although we have not tried and and do not guarantee it.  Note that Macs have recently not been equipped with NVIDIA GPUs so that using a GPU to execute this code on a Mac is not feasible.  It is possible to run the code on a CPU although it is slower.  The code was run with Python 3.8.8 within an Anaconda environment that contained Tensorflow 2.7.  

- The Anaconda distribution can be downloaded from this site: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- An environment file is located in this folder that can be used to create an Anaconda environment for this code.  
  - `tf2.yml`
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open an Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt with the environment of your choice:
    - <pre><code>conda env create -f <em>path_to_file</em>/tf2.yml</code></pre>

For a GPU to be successfully used, it must be `cuda` enabled and, furthermore, graphics drivers must be up to date with the appropriate versions of `cuda` and `cudnn`  installed.   Success with a GPU is also dependent, of course, on the GPU having sufficient memory.

## Executing the Code

Adversarial examples can be generated in two modes: 
- A "worker" program can be executed from the command line to generate one adversarial for a specified CIFAR-10 image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times for a sequence of MNIST images from the list of 500 randomly selected MNIST IDs in `../input/randMNIST.json`.  The controller program uses multiprocessing.

### Using the Worker Program

The filename for the worker program is `ga_cifar_worker.py`, which is located in the `code` folder.  It can be run from the command line or by specifying command line arguments in an IDE, for example, in Anaconda Spyder by choosing ``Run>Configuration per file`` and then specifying input arguments in the ``Command line options`` dialog field.

The input arguments for these command line programs are as follows, in this order:
- `cifar_id`: an integer from 0 to 59,999
- `model_file`: `model2.h5`
- `out_folder`: ../output/ (Output may be directed to another filepath if desired)
- 'in_folder`: ../input/  (do not change this specification)
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin`
- `select_type`: one of `proportionate`, `rank-linear`, or `rank-nonlinear`
- `rand_type`: `bright` for brightness mutation
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `gpu_mode`: either `FF.json` or `CNN.json`

Notes on input arguments:
- There are fewer input arguments for the CIFAR-10 code compared with the MNIST code because the parameters that were optimized during the initial test runs and that were invariant in the experiments were hard-coded indto the program file.
- If `gpu_mode` is `False`, then `tensorflow 2.0` will automatically use multiprocessing on however many CPU cores are available.

The article reports that the population was initialized with batches of candidate adversarial examples that were generated in batch sizes of 50,000.  While this was the case for the adversarial examples reported on in the article, the code in this repository usese batch sizes of 5,000 to reduce memory consumption.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):

><code>python <em>file_path_to_code</em>/ga_cifar_worker.py 0 model2.h5 ../output/ ../input/ L2 rank-linear bright 0.9 False</code>

Output from this code includes the following items:
- In folder `output/images`:
  - <code><em>scen_i</em>.npy</code>: a `numpy` file containing the adversrial example for scenario name as specified in the input arguments for CIFAR-10 image `i`.



### Using the Controller Program

The controller program filename is `ga_cifar_control.py` and it executes multiple worker files in parallel.  Its input arguments are:
- `start_id`: an integer from 0 to 59,999 representing the first CIFAR-10 iamge for which an adversarial example is generated
- `end_id`: an integer from 0 to 59,999 representing the last CIFAR-10 iamge for which an adversarial example is generated
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin, mad-recip, mad-linear`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `num_proc`: number of processors to be used in parallel mode
- `file_model`: `model2.h5`
- `folder`: `../input`, input folder
- `folder_out`: `../output``, output folder
- `gpu_mode`: `True` to use GPU and `False` otherwise.
- `mp_mode`: `True` if multiprocessing mode is used, and `False` otherwise.  Must be `False` if `gpu_mode = True`.
- `batch_id`: used to identify the batch of CIFAR-10 IDs being generated

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><code>python <em>file_path_to_code</em>/ga_cifar_control.py 0 19 L2 rank-linear bright 0.9 10 model2.h5 ../input/ ../output/ True False 0</code>

Output from this code includes the following items:
- In folder `output/images`:
  - <code><em>scen_i</em>.npy</code>: a `numpy` file containing the adversarial example for scenario name `scen`, which is created by the controller program to be the parameters joined with underscores, for CIFAR-10 image  `i`.
- In folder `output`:
  - `L2_rank-linear_bright_0.9__0_1.csv`, for example, where `L2_rank-linear_bright_0.9__` specifies the genetic algorithm parameters and, here, `0_1` indicates the starting and ending CIFAR-10 indices.
  - `0.csv` where, here, `0` stands for the `batch_id` specified in the input arguments. This file contains these output fields:
    - Scenario name
    - CIFAR-10 ID
    - Reference CIFAR-10 image classification
    - Classification of best adversarial example
    - Execution time to initialize population
    - Total Elapsed time
    - A string representing the adversarial example

