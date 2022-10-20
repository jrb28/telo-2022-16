This folder contains the code for the generation of adversarial examples for the MNIST data set.

# Computing Environment

We provided below directions to run the MNIST AE code in an Anaconda environment, as we did. The environment included `Python 3.7.7` and `Tensorflow 1.14`.  This particular version of `tensorflow` was used to be consistent with existing code for the Projected Gradient Descent (PGD)[^pgd] method.

- The Anaconda distribution can be downloaded from this site: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)
- Two environment files are located in this folder that can be used to create an Anaconda environment for this code.  One is for using the computer's CPU and the other the GPU if the computer is equipped with a cuda-enabled graphical processing unit.
  - `keras-cpu_tf14.yml`
  - `keras-gpu_tf14.yml`
- To install the environment:
  - First, install a base Anaconda environment from the URL above
  - Next, open an Anaconda Command Prompt (as Administrator)
  - Finally, execute this command in the Anaconda Command Prompt with the environment of your choice:
    - <pre><code>conda env create -f <em>path_to_file</em>/keras-gpu_tf14.yml</code></pre>

For a GPU to be successfully used, it be cuda enabled and the appropriate versions of `cuda` and `cudnn` must be installed. 





[^pgd]: xxxxx
