# MNIST Adversarial Examples

Folder contents include:
- Adversarial example files in `numpy` format.  Filenames reflect the parameters used to generate the adversarial examples.  The file named `L2-lin_rank-linear.npy`, for example, contains adversarial images generated with the linear form of the L2 fitness function and the reank-linear selection method. 
- `mnist_rand_images.npy`: a `numpy` file containing the CIFAR-10 images corresponding with the adversarial examples in the files above in a 
- `rand_mnist_indices`: a file with the randomly selected `CIFAR-10` image indices for which adversarial examples were generated
