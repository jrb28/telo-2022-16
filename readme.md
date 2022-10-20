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

Code is provided in this repository for generating adversarial examples for images from the MNIST[^mnist] and CIFAR-10[^cifar] data sets as described in the two subsequent subsections.

## MNIST

Adversarial examples can be generated in two modes: 

- A "worker" program can be executed from the command line to generate one adversarial for a specified MNIST image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times for a sequence of MNIST images.  The controller program uses multiprocessing.

The `code` and `input` folders are located in the `MNIST` folder in this root folder.

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
- `file_model`: either `FF.json` or `CNN.json` (Note: the corresponding weights file must be used.)
- `file_weights`: either `FF.h5` or `CNN.h5` (Note: the corresponding model file must be used.)
- `out_folder`: ../input
- `in_folder`: ../output
- `pop_size`: an integer representing the population size
- `prob_mut_genome`: probability that an image is mutated (0.7 in our experiments
- `prob_mut_pixel`: the probability of any pixel being mutated, if an image is mutated (0.00255 in our experiments)
- `num_gen`: an integer representing the number of generations to be executed
- `scen_name`: A scenario name to be associated with this set of parameters.

Notes on these parameters:
- The `folder_in` specified is consistent with the file folder structure in this repo and it permits the appropriate input files to be found.  Do not use a different argument.
- The `folder_out` specified above is also consistent with the folder structure in this repo, although a different output path could be specified if desired.
- The `scen_name` argument is a label to be associated with the particular parameters being used.  All results are put in the folder at this path `out_folder/scen_name`.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_mnist_adv_worker_rank-sel.py 0 L2 rank-linear rand 0.9 FF.json FF.h5 xxx xxx 1000 1.0 xxx  2000 0_L2_rl_rand_FF_1000_2000<\code></pre>
A similar command can be used with `ga_mnist_adv_worker_rank-sel_cnn.py` although the neural network files, `CNN.json` and `CNN.h5`, would be referenced in the input arguments.

Output from this code includes the following files in the designated output folder where _i_ is the `mnist_id`:
- <code><em>i</em>_elite.csv</code> -  Records the fitness of the elite phenotypes that are carried over into the next generation.  Each row contains, in this order, the MNIST ID, the generation number, the classification of the image by the neural network, and the phenotype's fitness.
- <code><em>i</em>_elite_parents.csv</code>: a file where each row has these elements: MNIST ID, generation number, number of offspring whose classifications match the ground truth label of the reference image, which necessitates that one of the offspring's parents be retained in the population for the next generation.
- <code><em>i</em>_img.npy</code>: a `numpy` file with the adversarial example with the greatest fitness.
- <code><em>i</em>_pop_stat</code>: a file where each row has these elements: MNIST ID, generation number, maximum population fitness, minimum fitness, mean fitness, median fitness


### Using the Controller Program

The controller program filename is `ga_control_rank_sel.py` and it executes multiple worker files in parallel.  Its input arguments are:
- `start_id`: an integer from 0 to 59,999 representing the first MNIST iamge for which an adversarial example is generated
- `end_id`: an integer from 0 to 59,999 representing the last MNIST iamge for which an adversarial example is generated
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `mp_mode`: (multiprocessing mode)  multiprocessing is used if `True` and not if `False`.  This argument must be set to `False` if a GPU is used because GPUs do not support multiprocessing.  If `True`, then `num_proc` argument is ignored.
- `num_proc`: number of processors to be used in parallel mode (_Note: a value greater than one cannot be used with a GPU as GPUs do not support multiprocessing.  Using `num_proc` > 1 works only with multiple-core CPUs._)
- `file_model`: either `FF.json` or `CNN.json`
- `file_weights`: either `FF.h5` or `CNN.h5`
- `folder_in`: ../input
- `folder_out`: ../output
- `filepath_code`: the filepath (and filename) for the pyrrhon executable worker file `ga_mnist_adv_worker_rank-sel.py` or `ga_mnist_adv_worker_rank-sel_cnn.py`
- `pop_size`: an integer representing the population size
- `prob_mut_genome`: probability that an image is mutated (0.7 in our experiments
- `pixel_mut_per_phenotype`: the expected number of pixels to be mutated in an image, if an image is mutated (2.0 in our experiments)
- `num_gen`: an integer representing the number of generations to be executed

Notes on these parameters:
- See notes above for teh worker program regarding the arguments `folder_in` and `folder_out`.
- The controller program creates the `scen_name` argument that is used by the worker program as a folder name for the output.  It is a combination of the GA parameters being used.  

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_control_rank_sel.py 0 19 L2 rank-linear rand 0.9 10 FF.json FF.h5 xxx xxx xxx 1000 1.0 xxx  2000</code></pre>

The same output is generated with the controller as would be generated with the worker program used to create a single adversarial example. 




## CIFAR-10

Adversarial examples can be generated in two modes: 
- A "worker" program can be executed from the command line to generate one adversarial for a specified CIFAR-10 image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times for a sequence of MNIST images.  The controller program uses multiprocessing.

The `code` and `data` folders are located in the `CIFAR` folder in this root folder.  The `data` folder contains input files.

### Using the Worker Program

The worker program is entitled: `ga_cifar_worker.py`.  It can be run from the command line or by specifying command line arguments in an IDE, for example, in Anaconda Spyder by choosing ``Run>Configuration per file`` and then specifying input arguments in the ``Command line options`` dialog field.

The input arguments for these command line programs are as follows, in this order:
- `cifar_id`: an integer from 0 to 59,999
- `model_file`: `model2.h5`
- `out_folder`: 
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `gpu_mode`: either `FF.json` or `CNN.json`

There are fewer input arguments for the CIFAR-10 code compared with the MNIST code because the parameters taht were optimized during the initial test runs and that were invariant in the experiments were hard-coded indto the program file.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_cifar_worker.py 0 xxx xxx L2 rank-linear rand 0.9 xxx<\code></pre>


Output from this code includes the following items:
- xxxx


### Using the Controller Program

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



[^mnist]: See xxx and xxx
[^cifar]: See xxx
