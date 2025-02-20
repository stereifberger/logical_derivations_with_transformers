# Define number of propositional variables
t_nu = 5

# PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.nn.functional import pairwise_distance

import losses
from functools import partial
from multiprocessing import Pool, cpu_count
from torch.utils.data import DataLoader, random_split

# NumPy
import numpy as np
from numpy import argmax

# Other libraries
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from random import seed, randint, choice, shuffle, sample, random
import pandas as pd
import copy
import re
import importlib
import statistics
import calculi, util, generation, architectures, losses, schedule
from torch import Tensor
from ipywidgets import IntText
from IPython.display import display
import time
import math
import contextlib
import io
from IPython.display import display
from IPython.utils import io
import json
from matplotlib import font_manager as fm
from torch.nn.functional import pairwise_distance