'''
================================================

Config file to hold all global and local configuration

Author : Abhishek Srivastava

================================================
'''

import torch
from dataclasses import dataclass


@dataclass
class CONFIG:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')