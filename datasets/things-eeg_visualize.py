# This Script is used to visualize the EEG data from the THINGS dataset.
# It uses the MNE library to plot the EEG data.

import mne
import numpy as np
import matplotlib.pyplot as plt
import torch
import io
from PIL import Image

# Data path (specify path to subj-01 as an example)
data_path = 'datasets/THINGS-EEG/Preprocessed_data_250Hz_whiten/sub-01'
img_set_path = 'datasets/THINGS-EEG/Image_set_Resize'

# We have two file under this data path:
# NOTE: these files are saved by torch.save
# 1.test.pt
# 2.train.pt

# -------------- Load the data -------------- #

# Load the data
train_data = torch.load(f'{data_path}/train.pt', weights_only = False)
test_data = torch.load(f'{data_path}/test.pt', weights_only = False)

# Get the types of the data -> Dict
print(f"Type of train_data: {type(train_data)}")
print(f"Type of test_data: {type(test_data)}")

# Get the keys of the data
# -> dict_keys(['eeg', 'label', 'img', 'text', 'session', 'ch_names', 'times'])
print(f"Keys of train_data: {train_data.keys()}")
print(f"Keys of test_data: {test_data.keys()}")

# Show each key's type and shape
for key in train_data.keys():
    # if the value has shape attribute, print the shape, otherwise print the length
    if hasattr(train_data[key], 'shape'):
        print(f"Key: {key}, Type: {type(train_data[key])}, Shape: {train_data[key].shape}")
    else:
        print(f"Key: {key}, Type: {type(train_data[key])}, Length: {len(train_data[key])}")
# Key: eeg, Type: <class 'numpy.ndarray'>, Shape: (16540, 4, 63, 250)
# Key: label, Type: <class 'numpy.ndarray'>, Shape: (16540, 4)
# Key: img, Type: <class 'numpy.ndarray'>, Shape: (16540, 4)
# Key: text, Type: <class 'numpy.ndarray'>, Shape: (16540, 4)
# Key: session, Type: <class 'numpy.ndarray'>, Shape: (16540, 4)
# Key: ch_names, Type: <class 'list'>, Length: 63
# Key: times, Type: <class 'numpy.ndarray'>, Shape: (300,)

# The training set includes 1654
# concepts with each concept 10 images, and each image re
# peats 4 times (1654 concepts * 10 images/concept * 4 trials/image) 
# per subject. 

# -------------- Visualization -------------- #

# let's get one sample
sample_index = 0
# EEG data
EEG_data = train_data['eeg'][sample_index]  # shape: (4, 63, 250)
times = train_data['times']  # shape: (300,) -> start from -0.2s to 1s, with 250Hz sampling frequency, so we have 300 time points
ch_names = train_data['ch_names']  # list of channel names
sfreq = 250  # sampling frequency
# Label
label_sample = train_data['label'][sample_index]  # shape: (4,)
# GT image (path and label)
img_path: np.ndarray[str] = train_data['img'][sample_index]  # shape: (4,)
class_name: np.ndarray[str] = train_data['text'][sample_index]  # shape: (4,)
# Session
session_sample = train_data['session'][sample_index]  # shape: (4,)

# Get img path and read the GT image
img_path = img_path[0]  # get the first trial's img path
# Append the img_set_path to the img_path
img_path = f"{img_set_path}/{img_path}"
# Read the image
with open(img_path, 'rb') as f:
    img = Image.open(io.BytesIO(f.read()))

# Print label / session
print(f"Label: {label_sample}, \nSession: {session_sample}")

# Define the figure and axes
fig, axs = plt.subplots(1, 2, figsize = (12, 6), dpi = 100)
# Plot EEG data
# prepare the info for mne
info = mne.create_info(ch_names = ch_names, sfreq = sfreq, ch_types = 'eeg')
# create the Evoked object
evoked = mne.EvokedArray(EEG_data.mean(axis = 0), info, tmin = 0)
# plot the evoked data
# (each channel's average over 4 trials) on the first axis
# Plot each channel individually
evoked.plot(spatial_colors = 'auto', 
            axes = axs[0], 
            show = False)
axs[0].set_title('EEG Data')
# Plot image with label as title
axs[1].axis('off')
axs[1].set_title(f"GT Image: {class_name[0]}")
axs[1].imshow(img)
plt.tight_layout()
# Save the figure
plt.savefig('datasets/things-eeg_visualize.png')
