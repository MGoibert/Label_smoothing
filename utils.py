"""
Author: Elvis Dohmatob <gmdopp@gmail.com>
"""

import sys
import logging
import argparse

from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F

# config
torch.set_default_tensor_type(torch.DoubleTensor)
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def parse_cmdline_args():
    """
    Parses command-line arguments / options for this software.
    """
    parser = argparse.ArgumentParser(
        description="Run experiments like a boss!",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--batch_size', type=int, default=100,
        help="batch size for SGD")
    parser.add_argument(
        '--num_epochs', type=int, default=7,
        help="number of passes to make over data")
    parser.add_argument(
        '--learning_rate', type=float, default=1e-4,
        help="learning rate for SGD")
    parser.add_argument(
        '--use_cuda', action="store_true",
        help="whether to use cuda or not")
    parser.add_argument(
        '--num_epsilons', type=int, default=7,
        help='number of epislon values to consider in attack')
    parser.add_argument(
        '--num_jobs', type=int, default=1,
        help="number of jobs to spawn for the experiment")

    return parser.parse_args()

