from io import StringIO
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import csv

with np.load('temperature_data.npz',allow_pickle=True) as f:
    # Get the existing data
    existing_data = f['data']
    print(existing_data[0:10])
h=torch.load('h_tensor.pt', map_location = torch.device("cuda"))
print(h[0])
print("ikinci: ")
print(h[1])