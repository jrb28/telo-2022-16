# Code for CIFAR-10 Adversarial Examples

| File | Description |
|:-------------------------:|:------------------------------------------------------------------------------------- |
| <code>ga_cifar_control.py </code> | Controller program to execute <code>ga_cifar_worker.py</code> multiple times for a sequence of CIFAR-10 IDs from `../input/randCIFAR.json`.|
| <code>ga_cifar_worker.py</code> | Worker program to generate one CIFAR-10 adversarial example |
| <code>view_compare_cifar.py</code> | View an adversarial example and its CIFAR-10 reference image.  |
| <code>compare_ae_cifar_file.py</code> | A program to compare a CIFAR-10 reference image with its adversarial example provided in the <code>numpy</code> files representing the adversarial examples reported on in the article. |

# Executing the Worker Program

The input arguments for <code>ga_cifar_worker.py</code> are:
- <code>cifar_id</code>: The CIFAR-10 ID for which an adversarial example is to be generated
- <code>model_file</code>: The neural network model file (<code>model2.h5</code>)
- <code>out_folder</code>: <code>../output/</code>
- <code>folder</code>: Input folder <code>../input/</code>
- <code>fit_type</code>: The fitness function; <code>L1, L2, Linf, L1-lin, L2-lin, Linf-lin</code>
- <code>select_type</code>: Selection operator; <code>proportionate, rank-linear, rank-nonlinear</code>
- <code>rand_type</code>: Mutation operator;  only <code>bright</code> is used for HSV brightness mutation
- <code>factor_rank_nonlinear</code>: Nonlinear rank selection factor (must be present even if alternative selection method is used)
- <code>gpu_mode</code>: Use GPU if <code>True</code> and not otherwise

A sample command line statement to execute the worker program is as follows:
><code>python <em>path_to_file/</em>ga_cifar_worker.py 0 model2.h5 ../output/ ../input/ L2 rank-linear bright 0.9 False</code>



# Executing the Controller Program

The input arguments for <code>ga_cifar_control.py</code> are:
- <code>start_id</code>: The initial CIFAR-10 ID in a sequence for which adversarial examples are to be generated
- <code>end_id</code>: The final (inclusive) CIFAR-10 ID in a sequence for which adversarial examples are to be generated
- <code>fit_type</code>: The fitness function; <code>L1, L2, Linf, L1-lin, L2-lin, Linf-lin</code>
- <code>select_type</code>: Selection operator; <code>proportionate, rank-linear, rank-nonlinear</code>
- <code>rand_type</code>: Mutation operator;  only <code>bright</code> is used for HSV brightness mutation
- <code>factor_rank_nonlinear</code>: Nonlinear rank selection factor (must be present even if alternative selection method is used)
- <code>num_proc</code>: Number of processor cores to be used in parallel execution
- <code>file_model</code>: The neural network model file (<code>model2.h5</code>)
- <code>folder</code>: Input folder <code>../input/</code>
- <code>folder_out</code>: <code>../output/</code>
- <code>gpu_mode</code>: Use GPU if <code>True</code> and not otherwise
- <code>mp_mode</code>: Use multi-processing if <code>True</code> and not otherwise,
- <code>batch_id</code>: An ID associated with the execution of this sequence of CIFAR-10 adversarial examples

A sample command line statement to execute the worker program is as follows:
><code>python <em>path_to_file/</em>ga_cifar_control.py 0 19 L2 rank-linear bright 0.9 11 model2.h5 ../input/ ../output/ True False x</code>

