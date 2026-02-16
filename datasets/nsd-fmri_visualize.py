# NOTE: the download dataset is packed in webdataset format,
# which is a collection of tar files, each containing a subset of the data.

import os
import io
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path to dataset
dataset_path = 'datasets/NSD/webdataset_avg_new'
# In this directory, we have
# - test
#     - test_subj01_0.tar
#     - test_subj01_1.tar
#     - test_subj02_0.tar
#     - test_subj02_1.tar
#     - ...
# - train
# - val
# - metadata_subj01.json
# - metadata_subj02.json
# - ...

# Try to get the first and second tar file in the test set
# should be test_subj01_0.tar and test_subj01_1.tar, 
# which contain the data for subject 1. 
# (0, 1 is webdataset's way of splitting the data into multiple 
# tar files for easier handling)
test_dir = os.path.join(dataset_path, 'test')
tar_files = [f for f in os.listdir(test_dir) if f.endswith('.tar')]
if not tar_files:
    raise FileNotFoundError("No tar files found in the test directory.")
tar_files.sort()  # Sort to ensure consistent order
first_tar_path = os.path.join(test_dir, tar_files[0])
second_tar_path = os.path.join(test_dir, tar_files[1])

# Open the first 2 tar files and list its contents
with tarfile.open(first_tar_path, 'r') as tar:
    print(f"Opening tar file: {first_tar_path}")
    members = tar.getmembers()
    print(f"Number of members in the tar file: {len(members)}")
    for member in members[:10]:  # Print first 10 members
        print(member.name)

with tarfile.open(second_tar_path, 'r') as tar:
    print(f"Opening tar file: {second_tar_path}")
    members = tar.getmembers()
    print(f"Number of members in the tar file: {len(members)}")
    # for member in members[:10]:  # Print first 10 members
    #     print(member.name)

# for each sample we have the following files:
# sample000000354.coco73k.npy       -> contain coco image id
# sample000000354.jpg               -> the image itself
# sample000000354.nsdgeneral.npy    -> contain the fMRI data for this sample (trial, voxel)
# sample000000354.num_uniques.npy   -> (forget about this) num of trials this image was shown to the subject most likely 3
# sample000000354.trial.npy         -> the ith image the subject saw

# Let's try to read one sample (5 files)
# We dynamically get the sample id
print(f"\nReading the first sample from the first tar file: {first_tar_path}")
with tarfile.open(first_tar_path, 'r') as tar:
    # Get members
    members = tar.getmembers()
    # Sort members to ensure consistent order
    members.sort(key=lambda x: x.name)
    # Get the first sample (5 files)
    member = members[ 0 : 5 ]  # Get the first 5 members (5 files for one sample)
    sample_id = member[0].name.split('.')[0]  # Get the sample id from the file name
    print(f"-> Sample ID: {sample_id}")
    
    # Read the 5 files for this sample
    coco_path = f"{sample_id}.coco73k.npy"
    img_path = f"{sample_id}.jpg"
    nsdgeneral_path = f"{sample_id}.nsdgeneral.npy"
    num_uniques_path = f"{sample_id}.num_uniques.npy"
    trial_path = f"{sample_id}.trial.npy"
    
    coco_data = np.load(io.BytesIO(tar.extractfile(coco_path).read()))
    img_data = np.array(Image.open(io.BytesIO(tar.extractfile(img_path).read())))
    nsdgeneral_data = np.load(io.BytesIO(tar.extractfile(nsdgeneral_path).read()))
    num_uniques_data = np.load(io.BytesIO(tar.extractfile(num_uniques_path).read()))
    trial_data = np.load(io.BytesIO(tar.extractfile(trial_path).read()))
    
    print(f"-> Coco data shape: {coco_data.shape}")
    print(f"-> Image data shape: {img_data.shape}")
    print(f"-> NSD general data shape: {nsdgeneral_data.shape}")
    print(f"-> Num uniques data shape: {num_uniques_data.shape}")
    print(f"-> Trial data shape: {trial_data.shape}")
    # Coco data shape: (1,)
    # Image data shape: (224, 224, 3)
    # NSD general data shape: (3, 15724)
    # Num uniques data shape: (1,)
    # Trial data shape: (1,)

    # Print coco data
    print(f"-> COCO data: {coco_data}")
    # Print NSD general data
    print(f"-> NSD general data: {nsdgeneral_data}")
    # Print num uniques data
    print(f"-> Num uniques data: {num_uniques_data}")
    # Print trial data
    print(f"-> Trial data: {trial_data}")
    # -> COCO data: [2950]
    # -> NSD general data: [[ 0.7554  0.1214  0.376  ... -0.1268 -0.2646  0.0185]
    #  [ 0.2278  0.3733 -0.2041 ... -0.6064  0.879   0.852 ]
    #  [ 0.4226 -1.517  -0.493  ...  0.2522  0.9585 -0.382 ]]
    # -> Num uniques data: [3]
    # -> Trial data: [2615]

    # Visualize the image
    plt.figure(figsize = (10, 5), dpi = 100)
    plt.imshow(img_data)
    plt.title("Image")
    plt.savefig("datasets/nsd-fmri_visualize.png")