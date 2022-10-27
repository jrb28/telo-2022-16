# CIFAR-10 Adversarial Examples

Folder contents include:
- Adversarial example files in `numpy` format.  Filenames reflect the parameters used to generate the adversarial examples.  The file named `L2-lin_rank-linear.npy`, for example, contains adversarial images generated with the linear form of the L2 fitness function and the rank-linear selection method.  Each file contains 500 adversarial examples generated for the 500 randomly selected CIFAR-10 IDs.
- `cifar_rand_images.npy`: a `numpy` file containing the 500 randomly selected CIFAR-10 images corresponding with the adversarial examples in the files above 
- `rand_cifar_indices`: a file with the 500 randomly selected `CIFAR-10` image indices for which adversarial examples were generated
