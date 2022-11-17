# Code for MNIST Adversarial Examples

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>ga_control_rank_sel.py</code> | Controller program to execute either <code>ga_mnist_adv_worker_rank-sel.py</code> or <code>ga_mnist_adv_worker_rank-sel_cnn.py</code> multiple times for a sequence of MNIST IDs from the file `../../input/randMNIST.json`. |
| <code>ga_mnist_adv_worker_rank-sel.py</code> | Worker program to generate one adversarial example using the FF neural network |
| <code>ga_mnist_adv_worker_rank-sel_cnn.py</code> | Worker program to generate one adversarial example using the CNN neural network |
| <code>view_compare.py</code> | Code to view an adversarial example and its MNIST reference image |

The <code>adv_egs</code> folder contains the adversarial examples analysed in the paper, which were generated with the genetic algorithm.

An <code>output</code> folder has been provided to receive the adversarial example files.

Executing <code>ga_control_rank_sel.py</code> to cause adversarial examples to be generated for the 500 randomly selected MNIST IDs is done, for example, from the command line using this command

><code>python ga_control_rank_sel.py </code>

Please see the code for a description of the arguments or, alternately execute this command from teh command line:

><code>python python ga_control_rank_sel.py -h</code>

Command line arguments can, alternatively, be specified in some IDEs.
