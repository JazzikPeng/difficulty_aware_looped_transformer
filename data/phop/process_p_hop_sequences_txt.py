"""
This file is used to generate phop tasks. 
Organize pHop task into a next token generation task.
"""

# Read from /home/jupyter/project/nanoGPT/data
# /home/jupyter/project/nanoGPT/data
import os
import time
import math
import pickle
import torch
import numpy as np

# Read txt files and pad them to the same length 
