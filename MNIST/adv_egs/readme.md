# MNIST Adversarial Examples

Folder contents include:
- Adversarial example files in `numpy` format.  Filenames reflect the parameters used to generate the adversarial examples.  The file named `FF_L2_rank-linear_rand.npy`, for example, contains adversarial images generated with the feedforward (FF) neural network, the L2 fitness function, the rank-linear selection method, and random mutation. Each file contains 500 adversarial examples generated for the 500 randomly selected MNIST IDs.
- `mnist_rand_images.npy`: a `numpy` file containing the CIFAR-10 images corresponding with the adversarial examples in the files above in a 
- `rand_mnist_indices`: a file with the 500 randomly selected `CIFAR-10` image indices for which adversarial examples were generated
