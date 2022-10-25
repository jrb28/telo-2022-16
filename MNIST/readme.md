# MNIST Adversarial Examples 
This folder contains the code for the generation of adversarial examples for the MNIST data set.

## Subfolders in this Repository

- `code`: code for generating adversarial examples
- `input`: input files required to run the code
- `output`: folder for receiving output
- `adv_egs`: a static folder that contains the adversarial examples that were reported in in the article.  A `numpy` file is included in this folder for every parameter set that was evaluated and the filename is shorthand for the genetic algorithm parameters separated by underscores.

## Computing Environment

We provide below directions to run the MNIST AE code in an Anaconda environment, as we did. The environment includes `Python 3.7.7` and `Tensorflow 1.14`.  This particular version of `tensorflow` was used to be consistent with existing code for the Projected Gradient Descent (PGD)[^pgd] method.

- The Anaconda distribution can be downloaded from this site: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- Two environment files are located in this folder that can be used to create an Anaconda environment for this code.  One is for using the computer's CPU and the other the GPU, the latter of which can be used only if the computer is equipped with a cuda-enabled graphical processing unit.
  - `keras-cpu_tf14.yml`
  - `keras-gpu_tf14.yml`
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open an Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt with the environment of your choice:
    - <pre><code>conda env create -f <em>path_to_file</em>/keras-gpu_tf14.yml</code></pre>

For a GPU to be successfully used, it must be cuda enabled and, furthermore, graphics drivers must be up to date with the appropriate versions of `cuda` and `cudnn`  installed.  Success with a GPU is also dependent, of course, on the GPU having sufficient memory.


## Executing the Code

Adversarial examples can be generated in two modes: 
- A "worker" program can be executed from the command line to generate one adversarial for a specified MNIST image.
- A "controller" program can be run from the command line to execute the "worker" program multiple times to generate adversarial examples for a sequence of MNIST images.  The controller program uses multiprocessing.


### Using the Worker Program

Two worker programs exist in the `code` folder, one for Feedforward Neural Networks (FFs) and a separate program for Convolutional Neural Networks (CNNs):
- FF: `ga_mnist_adv_worker_rank-sel.py`
- CNN: `ga_mnist_adv_worker_rank-sel_cnn.py`

These programs can be run from the command line or by specifying command line arguments in an IDE, for example, in Anaconda Spyder by choosing ``Run>Configuration per file`` and then specifying input arguments in the ``Command line options`` dialog field.

The input arguments for these command line programs are as follows, in this order:
- `mnist_id`: an integer from 0 to 59,999
- `fit_type`: one of `L1`, `L1-lin`, `L2`, `L2-lin`, `Linf`, `Linf-lin`, `mad-recip`, or `mad-linear`
- `select_type`: one of `proportionate`, `rank-linear`, or `rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.  Results reported in the article used 0.9.
- `file_model`: either `FF.json` or `CNN.json` (Note: the corresponding weights file and code file must be used .)
- `file_weights`: either `FF.h5` or `CNN.h5` (Note: the corresponding model file and code file must be used.)
- `out_folder`: ../output/ (this correspnds with repo folder structure, but this may be changed if desired)
- `in_folder`: ../input/ (do not deviate from this argument to suit repo folder structure)
- `pop_size`: an integer representing the population size.  The article reports results using 1000.
- `prob_mut_genome`: probability that an image is mutated (0.7 in our experiments)
- `prob_mut_pixel`: the probability of any pixel being mutated, if an image is mutated (0.00255 in our experiments)
- `num_gen`: an integer representing the number of generations to be executed (2000 in our experiments)
- `scen_name`: a scenario name to be associated with this set of parameters

Notes on these parameters:
- The `folder_in` specified is consistent with the file folder structure in this repo and it permits the appropriate input files to be found.  Do not use a different argument.
- The `folder_out` specified above is also consistent with the folder structure in this repo, although a different output path could be specified if desired.
- The `scen_name` argument is a label to be associated with the particular parameters being used.  All results are put in the folder at this path `out_folder/scen_name`.

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environment variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_mnist_adv_worker_rank-sel.py 0 L2 rank-linear rand 0.9 FF.json FF.h5 ../output/ ../input/ 1000 0.7 0.00255  2000 0_L2_rl_rand_FF_1000_2000<\code></pre>
A similar command can be used with `ga_mnist_adv_worker_rank-sel_cnn.py` although the neural network files, `CNN.json` and `CNN.h5`, would be referenced in the input arguments.

The following files are output in the indicated folders where `i` is the MNIST index:
- In <code>../output/<em>scen_name</em></code> where <code><em>scen_name</em></code> is the scenario name:
  - <code><em>i</em>_elite.csv</code> -  Records the fitness of the elite phenotypes that are carried over into the next generation.  Each row contains, in this order, the MNIST ID, the generation number, the classification of the image by the neural network, and the phenotype's fitness.
  - <code><em>i</em>_elite_parents.csv</code>: a file where each row has these elements: MNIST ID, generation number, number of offspring whose classifications match the ground truth label of the reference image, which necessitates that one of the offspring's parents be retained in the population for the next generation.
  - <code><em>i</em>_img.npy</code>: a `numpy` file with the adversarial example with the greatest fitness.
  - <code><em>i</em>_pop_stat</code>: a file where each row has these elements: MNIST ID, generation number, maximum population fitness, minimum fitness, mean fitness, median fitness
- In <code>../output/</code>:
  - `FF_L2_rank-nonlinear_rand_5.csv`, for example, where `FF_L2_rank-nonlinear_rand` represents the parameters and the trailing integer is indexed each time the controller program is run for a particular scenario.  The output includes the scenario name, MNIST ID, the reference image classification, the adversarial example classification, fitness if `mad-linear` fitness function is used, fitness of best adversarial example, elapsed execution time, and a string representing the adversarial example.
  - Other files indicating execution time



### Using the Controller Program

The controller program filename is `ga_control_rank_sel.py` and it executes multiple worker files in parallel.  It generates adversarial examples for 500 randomly selected MNIST IDs whose indices are in this file: `../input/randMNIST.json`.  The controller's input arguments are:
- `start_id`: an integer from 0 to 499 representing the starting index from the 500 random MNIST IDs for which adversarial examples are generated
- `end_id`: an integer from 0 to 499 representing the last index from the 500 random MNIST IDs for which adversarial examples are generated (must be greater than or equal to `start_id`)
- `fit_type`: one of `L1, L1-lin, L2, L2-lin, Linf, Linf-lin`, `mad-recip`, or `mad-linear`
- `select_type`: one of `proportionate, rank-linear, rank-nonlinear`
- `rand_type`: either `rand` or `mad`
- `factor_rank_nonlinear`: a floating-point value greater than 0.0 but less than 1.0.  This is a required argument even if rank-nonlinear selection is not being used.
- `mp_mode`: (multiprocessing mode)  multiprocessing is used if `True` and not if `False`.  This argument must be set to `False` if a GPU is used because GPUs do not support multiprocessing.  If `False`, then `num_proc` argument is ignored.
- `num_proc`: number of processors to be used in parallel mode (_Note: a value greater than one cannot be used with a GPU as GPUs do not support multiprocessing.  Using `num_proc` > 1 works only with multiple-core CPUs._)
- `file_model`: either `FF.json` or `CNN.json`
- `file_weights`: either `FF.h5` or `CNN.h5`
- `folder_in`: ../input/
- `folder_out`: ../output/
- `filepath_code`: the filepath (and filename) for the python executable worker file.  If the folder structure of this repo is maintained, then use either `ga_mnist_adv_worker_rank-sel.py` or `ga_mnist_adv_worker_rank-sel_cnn.py`, whichever is consistent with the `file_model` and `file_weights` arguments. 
- `pop_size`: an integer representing the population size (1000 in our experiments)
- `prob_mut_genome`: probability that an image is mutated (0.7 in our experiments)
- `pixel_mut_per_phenotype`: the expected number of pixels to be mutated in an image, if an image is mutated (2.0 in our experiments)
- `num_gen`: an integer representing the number of generations to be executed (1000 in our experiments)

Notes on these parameters:
- See notes above for the worker program regarding the arguments `folder_in` and `folder_out`.
- The controller program creates the `scen_name` argument that is used by the worker program as a folder name for the output.  It is a combination of the GA parameters being used, joined with underscores.  

To execute from the command line, open a command  prompt that recognizes the path to the python executable (either an Anaconda command prompt or a Windows command prompt if the environemnt variables are set properly to find the python executable) and execute this command (with some example input arguments):
><pre><code>python <em>file_path_to_code</em>/ga_control_rank_sel.py 0 19 L2 rank-linear rand 0.9 10 FF.json FF.h5 xxx xxx xxx 1000 1.0 2  2000</code></pre>

The same output files are generated with the controller as would be generated with the worker program used to create a single adversarial example and, in addition, these files are created in the `../output/` folder:
- <code><em>folder_out</em>/CNN_1000_25_70_2_L2_rank-linear_rand_timing.txt</code>, for example, where in this case `CNN_1000_25_70_2_L2_rank-linear_rand` are the genetic algorithm parameters  joined by underscores.  This file documents execution time.
- <code><em>folder_out</em>/CNN_L2_rank-linear_rand_0.csv</code>, for example, where `CNN_L2_rank-linear_rand` indicate the genetic algorithm parameters and `0` is an index that is incremented by 1 each time a new batch is run for the same combination of parameters.  this file documents, in each row, the parameter set, the MNIST index, the reference MNIST label, the adversarial classification, the `mad-linear` fitness value if that fitness function is used, the fitness of the best adversarial example, the execution time, and a string indicating the MNIST adversarial example.
- <code><em>folder_out</em>/L2_rank-linear_rand_0.9__0_1.csv</code>, for example is execution time for MNIST images with indices starting with `0` and ending with `1`, where `L2_rank-linear_rand_0.9` are the genetic algorithm parameters.
- <code><em>scen_name</em>_timing.txt</code>, where *scen_name* is the scenario name.  The file contains the number of seconds required for generating adversarial examples for all of the multiple MNIST images. 



[^pgd]: See Madry, A.,  Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A., Accessed 2020, [https://github.comMadryLabmnist_challenge](https://github.comMadryLabmnist_challenge). and  Madry, A.,  Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2017) "Towards Deep Learning Models Resistant to Adversarial Attacks", [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083)
