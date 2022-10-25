# Code for CIFAR-10 Adversarial Examples

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>ga_cifar_control.py </code> | Controller program to execute <code>ga_cifar_worker.py</code> multiple times for a sequence of CIFAR-10 IDs |
| <code>ga_cifar_worker.py</code> | Worker program to generate one CIFAR-10 adversarial example |
| <code>view_compare_cifar</code> | View an adversarial example and its CIFAR-10 reference image.  Note that this program will present a comparison of corresponding CIFAR-10 reference images and adversarial examples only when the adversarial examples are generated through the controller program, <code>ga_cifar_control.py </code>. |
