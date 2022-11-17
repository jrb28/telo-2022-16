# Code for Fast Gradient Sign Method (FGSM)

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>fgsm_mgr.py</code> | This is a command-line program that executes <code>fgsm_wkr_simple.py</code> multiple times to generate a sequence of FGSM adversarial examples for the MNIST IDs included the file <code>../../input/randMNIST.json<\code>. |
| <code>fgsm_wkr_simple.py</code> | A command-line worker program that generates one adversarial example for a specified MNIST ID. |
| <code>gather_numpy_files.py</code> | A program to accumulate individual text files for each adversarial image and concatenate them into a single <code>numpy</code> file. |

An <code>output</code> folder has been provided to receive the adversarial example files.

The program <code>fgsm_mgr.py</code> can be executed from the command line or by specifying command line arguments in your IDE.  Executing this program from the command line with the feedforward (FF) neural network, for example, is done with this command:

<code>python fgsm_mgr.py ../../input/ output/ randMNIST.json FF.json FF.h5 fgsm_wkr_simple.py</code>
