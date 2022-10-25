# Code for MNIST Adversarial Examples

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>ga_control_rank_sel.py </code> | Controller program to execute either <code>ga_mnist_adv_worker_rank-sel.py</code> or <code>ga_mnist_adv_worker_rank-sel_cnn.py</code> multiple times for a sequence of MNIST IDs |
| <code>ga_mnist_adv_worker_rank-sel.py</code> | Worker program to generate one adversarial example using the FF neural network |
| <code>ga_mnist_adv_worker_rank-sel_cnn.py</code> | Worker program to generate one adversarial example using the CNN neural networ |
| <code>view_compare</code> | View an adversarial example and its MNIST reference image |
