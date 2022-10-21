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
    - <pre><code>conda env create -f <em>path_to_file</em>/keras-gpu_tf14.yml</code></pre>

For a GPU to be successfully used, it be `cuda` enabled and, furthermore, graphics drivers must be up to date with the appropriate versions of `cuda` and `cudnn`  installed.   Success with a GPU is also dependent, of course, on the GPU having sufficient memory.
