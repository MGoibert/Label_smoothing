"""
Author: Elvis Dohmatob <gmdopp@gmail.com>
"""

import sys
import logging
import argparse

# config
# torch.set_default_tensor_type(torch.DoubleTensor)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_cmdline_args():
    """
    Parses command-line arguments / options for this software.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments like a boss!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', type=str, choices=["MNIST", "CIFAR10"],
        default="MNIST", help="choose the datset")
    parser.add_argument(
        '--experiment_name', type=str, choices=["temperature", "all"],
        default="all", help="batch size for SGD")
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help="batch size for SGD")
    parser.add_argument(
        '--test_batch_size', type=int, default=1,
        help="batch size for testing")
    parser.add_argument(
        '--num_epochs', type=int, default=7,
        help="number of passes to make over data")
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help="learning rate for SGD")
    parser.add_argument(
        '--num_alphas', type=int, default=11,
        help="number of alphas to use in smoothing")
    parser.add_argument(
        '--min_epsilon', type=float, default=0.,
        help="minimum adversarial perturbation in experiments")
    parser.add_argument(
        '--max_epsilon', type=float, default=.25,
        help="maximum adversarial perturbation in experiments")
    parser.add_argument(
        '--num_epsilons', type=int, default=11,
        help="number of epsilons to use in smoothing")
    parser.add_argument(
        '--model', type=str, choices=["Linear", "LeNet", "ResNet"],
        default="Linear", help="choose which model to use")
    parser.add_argument(
        '--attack_method', type=str, choices=["FGSM", "DeepFool", "BIM", "CW", "triangular"],
        default="FGSM", help="which type of adversarial attack")
    parser.add_argument(
        '--num_jobs', type=int, default=1,
        help="number of jobs to spawn for the experiment")
    parser.add_argument(
        '--num_gammas', type=int, default=1,
        help="number of gammas for triangular experiment")
    parser.add_argument(
        '--num_samples', type=int, default=10000,
        help="number of samples for triangular experiment")
    parser.add_argument(
        '--num_iter_attack', type=int, default=10000,
        help="number of (outer-most) iterations for iterative attacks")

    return parser.parse_args()