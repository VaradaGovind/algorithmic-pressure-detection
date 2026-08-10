import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data_dir = 'data/raw/WESAD'
subject = 'S2'
subject_path = os.path.join(raw_data_dir, subject, f'{subject}.pkl')

print(f"Loading data for {subject} from {subject_path}")

try:
    with open(subject_path, 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    print("Data loaded successfully!")
    
    print(f"\nKeys in the dataset: {data.keys()}")
    
    # Inspect the Empatica E4 sensor data
    e4_data = data['signal']['wrist']
    print(f"\nEmpatica E4 Sensors recorded: {e4_data.keys()}")
    
    label = data['label']
    print(f"Labels shape: {label.shape} - Unique states: {np.unique(label)}")
    
except Exception as e:
    print(f"Error loading WESAD data: {e}")