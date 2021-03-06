# Label-Smoothing and Adversarial Robustness

This repository contains code to run Label Smoothing as a means to improve adversarial robustness for deep leatning, supervised classification tasks.
See the paper for more information about Label-Smoothing and a full understanding of the hyperparatemer

Supported datasets and NN architectures:
  - MNIST; MLP or LeNet
  - CIFAR10; LeNet or ResNet18
  - SVHN; LeNet

Supported methods:
  - Label-Smoothing: standard (SLS), adversarial (ALS), boltzmann (BSL), second-best (SBLS)
  - Adversarial training as defined in [this paper](https://arxiv.org/pdf/1412.6572.pdf)
  - Traditional training (no Label-Smoothing, no adversarial training, no regularization)
  
Supported attacks to test adversarial robustness:
  - FGSM (see [here](https://arxiv.org/pdf/1412.6572.pdf))
  - BIM (see [here](https://arxiv.org/pdf/1611.01236.pdf))
  - DeepFool (see [here](https://arxiv.org/pdf/1511.04599.pdf))
  - C&W (see [here](https://arxiv.org/pdf/1608.04644.pdf))
  
  
## Purposes
  
You can run the experiment choosing your dataset, NN architecture, Label-Smoothing method(s) and parameters, and attack method(s).
The code will save:
  - The models you trained as .pt files (if enabled)
  - The results (i.e. the adversarial accuracy for each class label and parameter implemented) as .csv files
    
    
## Usage

You can run the code by running the main file: run_label_smoothing.py
You can tune several items:
  - dataset: MNIST, CIFAR10, SVHN (default: MNIST) --> which dataset to use
  - model: Linear, LeNet, ResNet (default: Linear) --> which NN architecture to use
  - batch_size (default: 100) --> size of the batch size for the training set
  - test_batch_size (default: 1) --> size of the batch size for the test set
  - learning_rate (default: 0.0001) --> learning rate for stochastic gradient descent
  - num_epochs (default: 50) --> number of epochs required to train the model
  - num_jobs (default: 1) --> number of jobs for parallelization
  - to_save_model --> if used, will save the model after training
  - use_saved_model --> if used, will use a formely saved model and skip training
  - smoothing_method: standard, adversarial, boltzmann, second_best (default: standard) --> which LS method to use (several are possible)
  - num_alphas (default: 11) --> number of LS hyperparameter alpha to run
  - min_alpha (default: 0) --> minimum value for the LS hyperparameter alpha
  - max_alpha (default: 1) --> maximum value for the LS hyperparameter alpha
  - attack_method: FGSM, BIM, DeepFool, CW (default: FGSM) --> which attack to run (several are possible)
  - num_epsilons (default: 11) --> number of attack strenghts epsilon to run (for FGSM and BIM, it is an hyperparameter)
  - min_epsilon (default: 0) --> minimum value for epsilon
  - max_epsilon (default: 0.25) --> maximum value for epsilon
  - adv_training --> if used, will enable adversarial training
  - adv_training_param (default: 0.2) --> strenght parameter for adversarial training
  - adv_training_reg_param (default: 0.75) --> regularization parameter for adversarial training

### Examples

#### A very simple example
```ipyhton --pdb -- run_label_smoothing.py --to_save_model --smoothing_method standard adversarial boltzmann second_best```

This code will train a Linear (MLP) model on MNIST, using each of the four Label-Smoothing methods with 11 parameters alpha ranging from 0 to 1.
Once trained, all these models will be saved. To test adversarial robustness, FGSM attack will be used (with each of the default value for epsilon).
The accuracy results for each model and parameters will be exported.

#### Tuning training
```ipyhton --pdb -- run_label_smoothing.py --to_save_model --use_saved_model --dataset CIFAR10 --model ResNet --num_epochs 150 --batch_size 50 --test_batch_size 50```

This code will train a ResNet18 with standard Label-Smoothing (with default alpha values) on CIFAR10 dataset, with 150 epochs. The trained models will be saved, and if pre-trained models exist, they will be used (training skipped instead).
FGSM attack with default strenght values will be run too.


#### Tuning attacks
```ipyhton --pdb -- run_label_smoothing.py --to_save_model --use_saved_model --dataset MNIST --model LeNet --attack_method DeepFool CW --num_epsilon 1```

LeNet models with standard Label-Smoothing and default alpha values will be trained, and then DeepFool and CW attacks will be run.
These attacks do not implement a strenght (epsilon) parameter so we can set the number of epsilon to 1.
  
