'''
Utility file containing helper functions.
'''

import torch
import numpy as np
import random

def set_seed(seed=123):
    # Fix the seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)