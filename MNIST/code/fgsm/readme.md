# Code for Fast Gradient Sign Method (FGSM)

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>fgsm_mgr.py </code> | Controller program to execute <code>fgsm_wkr_simple.py</code> multiple times for a sequence of MNIST IDs from the file `../../input/randMNIST.json`. |
| <code>fgsm_wkr_simple.py</code> | Worker program to generate one adversarial example using the FF neural network |
| <code>gather_numpy_files.py</code> | A program to accumulate individual text files for each adversarial image and concatenate them into a sinlge <code>numpy</code> file |

An <code>output</code> folder has been provided to receive the adversarial example files.
