# MNIST Adversarial Examples 
This folder contains the code for the generation of adversarial examples for the MNIST data set.

## Subfolders in this Repository

- `code`: code for generating adversarial examples with the genetic algorithm and the Fast Gradient Sign Method (FGSM)
- `input`: input files required to run the code

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





[^pgd]: See Madry, A.,  Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A., Accessed 2020, [https://github.comMadryLabmnist_challenge](https://github.comMadryLabmnist_challenge). and  Madry, A.,  Makelov, A., Schmidt, L., Tsipras, D., and Vladu, A. (2017) "Towards Deep Learning Models Resistant to Adversarial Attacks", [https://arxiv.org/abs/1706.06083](https://arxiv.org/abs/1706.06083)
