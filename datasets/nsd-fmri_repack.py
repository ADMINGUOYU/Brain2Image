# This script is used to repack the fMRI data from the NSD dataset.
# we use CLIPModel from transformers to process the
# embedding of GT images.
# transformer package is required to run this script.
# run "pip install transformers" to install the package.

"""
Please use 'python -m datasets.nsd-fmri_repack' to run this script.
"""

# We'll save the processed data in pandas.DataFrame:
# (fmri / image_index / subject / split / sample_id / tar_path).
# We'll also index every unique images within the used coco dataset
# and save it in a separate pandas.DataFrame:
# (image_index / image_data / image_embedding /
#  clip_bigG_embeddings / vae_latents / cnx_features /
#  subject / split)
# WARNING: A special set of 1,000 images was viewed by all participants.
#          other images are all unique. if shared 'subject' would be 'shared'.

# Helpful things to know:
# COCO_73k_subj_indices.hdf5
# - Contains datasets or keys for each subject 
#   (e.g., subj01, subj02, ..., subj08).

# --------------- Start of configuration --------------- #

# import os
import os
# set transformers cache directory
os.environ['TRANSFORMERS_CACHE'] = 'datasets/transformers_cache'
os.environ['HF_HOME'] = 'datasets/transformers_cache'
os.environ['HF_HUB_CACHE'] = 'datasets/transformers_cache'

# import necessary libraries
import torch
import pandas as pd
import numpy as np
import h5py
import tarfile
import io
from transformers import CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL
from model.autoencoder.convnext import ConvnextXL
import open_clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Data path (root folder to all subjects / images)
data_path = 'datasets/NSD/webdataset_avg_new'
coco_image_indices_path = 'datasets/NSD/COCO_73k_subj_indices.hdf5'
processed_data_dir = 'datasets/processed'
vae_encode_ckpt_path = 'datasets/NSD/sd_image_var_autoenc_mindeye2.pth'
convnext_xl_ckpt_path = 'datasets/NSD/convnext_xlarge_alpha0.75_fullckpt.pth'
os.makedirs(processed_data_dir, exist_ok = True)

# we specify the subjects we want to process
subject_wanted = ['sub-01'] # 8 in total

# CUDA device configuration
CUDA = 0

# ---------------- End of configuration ---------------- #

def process_nsd_fmri_image_all_subjects():
    # we first need to get the 'potentially' shared 1,000 images:
    # (actually we get 907) - not all subject completed the scan
    print("Finding the shared 1,000 images across all subjects...")
    # Load the subject indices
    with h5py.File(coco_image_indices_path, 'r') as f:
        # Get indices for each subject (using list comprehension)
        # we only use lists from 1,2,5,7 (completed all sessions)
        subject_indices = [set(f[f'subj0{i}'][:]) for i in [1, 2, 5, 7]]
    # Find the intersection (indices present in all 8 subjects)
    shared_1000_indices = set.intersection(*subject_indices)
    # Convert from np.int64 to python int
    shared_1000_indices = set(int(idx) for idx in shared_1000_indices)
    # Convert to a sorted list
    shared_1000_indices = sorted(list(shared_1000_indices))
    # verbose
    print(f"Found {len(shared_1000_indices)} shared images.")
    print(f"Shared image indices: {shared_1000_indices[:10]}...")

    # Now let's try to open the folders of data
    print("Processing images and the data for each subject...")
    # remember, under data_path, we have the following structure:
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
    # NOTE: remember to convert 'subject' to standard format: sub-XX
    # NOTE: we process all subjects first and then select the wanted ones
    # NOTE: we will also cache the full-subject processed data.
    # NOTE: each subject might have multiple tar files
    device = f'cuda:{CUDA}' if torch.cuda.is_available() else 'cpu'
    # initialize CLIP model and processor
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    # initialize OpenCLIP ViT-bigG-14 for clip_bigG_embeddings
    clip_bigG_model, _, clip_bigG_preprocess = open_clip.create_model_and_transforms(
        'ViT-bigG-14', pretrained = 'laion2b_s39b_b160k'
    )
    clip_bigG_model = clip_bigG_model.to(device).eval()
    clip_bigG_model.visual.output_tokens = True
    # initialize Stable Diffusion VAE for vae_latents
    vae_model = AutoencoderKL(
        down_block_types = ['DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'],
        up_block_types = ['UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'],
        block_out_channels = [128, 256, 512, 512],
        layers_per_block = 2,
        sample_size = 256,
    )
    vae_model.load_state_dict(torch.load(vae_encode_ckpt_path, map_location = device))
    vae_model.requires_grad_(False)
    vae_model.eval()
    vae_model.to(device)
    # initialize ConvNeXt-XL for cnx_features
    cnx_model = ConvnextXL(path = convnext_xl_ckpt_path)
    cnx_model.requires_grad_(False)
    cnx_model.eval()
    cnx_model.to(device)
    # through every split (we collect all subject)
    all_subjects_data = []
    all_images_data = { }
    for split in ['train', 'val', 'test']:
        # verbose
        print(f"Processing split: {split}...")
        # get all tar files in the split folder
        split_folder = os.path.join(data_path, split)
        tar_files = [f for f in os.listdir(split_folder) if f.endswith('.tar')]
        # we sort the tar files (so each subject's different parts will stick togethe)
        tar_files.sort()
        # process each tar file
        for tar_file in tqdm(tar_files, desc = f"Processing {split} tar files"):
            # extract subject from the tar file name
            # e.g., test_subj01_0.tar -> sub-01
            subject = tar_file.split('_')[1].replace('subj', 'sub-')
            # verify the subject format
            assert subject.startswith('sub-') and len(subject) == 6, f"Unexpected subject format: {subject}"
            # open the tar file using h5py (since it's actually an hdf5 file)
            tar_path = os.path.join(split_folder, tar_file)
            # open the tar file and read its contents
            with tarfile.open(tar_path, 'r') as tar:
                # get all members in the tar file
                members = tar.getmembers()
                # we sort the members to ensure consistent order (so each sample's 5 files will stick together)
                members.sort(key = lambda x: x.name)
                # process each sample (5 files for each sample)
                for i in range(0, len(members), 5):
                    # get the 5 files for the current sample
                    sample_members = members[i : i + 5]
                    # get the sample id from the file name (e.g., sample000000354.coco73k.npy -> sample000000354)
                    sample_id = sample_members[0].name.split('.')[0]
                    # read the coco image index from the .coco73k.npy file
                    coco_index_file = [m for m in sample_members if m.name.endswith('.coco73k.npy')][0]
                    coco_index_data = tar.extractfile(coco_index_file).read()
                    coco_index = int(np.load(io.BytesIO(coco_index_data)).item())
                    # check if the coco index is in the shared 1000 images
                    is_shared_image = coco_index in shared_1000_indices
                    # read the fMRI data from the .nsdgeneral.npy file
                    fmri_file = [m for m in sample_members if m.name.endswith('.nsdgeneral.npy')][0]
                    fmri_data = tar.extractfile(fmri_file).read()
                    fmri = np.load(io.BytesIO(fmri_data))
                    # create a dictionary to store the processed data for this sample
                    # (fmri / image_index / subject / split / sample_id / tar_path)
                    sample_fmri_data = {
                        'subject': subject,
                        'split': split,
                        'image_index': coco_index,
                        'fmri': fmri,
                        'sample_id': sample_id,
                        'tar_path': tar_path
                    }
                    # append the sample data to the list of all subjects data
                    all_subjects_data.append(sample_fmri_data)

                    # now we process the image
                    # check if the image is shared and already been processed
                    if is_shared_image and coco_index in all_images_data:
                        # warn user if the image is not same split
                        if all_images_data[coco_index]['split'] != split:
                            print(f"Warning: Shared image {coco_index} is in different splits: {all_images_data[coco_index]['split']} vs {split}")
                        # skip 
                        continue
                    # read the image from the .jpg file
                    image_file = [m for m in sample_members if m.name.endswith('.jpg')][0]
                    image_data = tar.extractfile(image_file).read()
                    image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
                    image_numpy = np.array(image_pil)
                    image_tensor_float = (torch.tensor(np.array(image_pil).transpose(2,0,1)).float() / 255.0).unsqueeze(0).to(device) # shape (1, 3, H, W), range [0, 1]
                    # process the image using CLIPProcessor and CLIPModel to get its embedding
                    # also extract clip_bigG_embeddings, vae_latents, cnx_features
                    with torch.no_grad():
                        # CLIP-ViT-L/14 embedding
                        inputs = processor(images = image_numpy, return_tensors = 'pt').to(device)
                        image_embedding = model.get_image_features(**inputs).pooler_output.cpu().squeeze().numpy()
                        # image_embedding = model.get_image_features(**inputs).cpu().squeeze().numpy()
                        
                        # OpenCLIP ViT-bigG-14 token embeddings (256 tokens, 1664 dim)
                        clip_bigG_input = clip_bigG_preprocess(image_pil).unsqueeze(0).to(device)
                        _ , clip_bigG_tokens = clip_bigG_model.visual.forward(clip_bigG_input)
                        clip_bigG_emb = clip_bigG_tokens.cpu().squeeze(0).numpy()
                        
                        # Stable Diffusion VAE latent encoding
                        vae_latent = vae_model.encode(2 * image_tensor_float - 1).latent_dist.mode().cpu().squeeze(0).numpy() * 0.18215
                        
                        # ConvNeXt-XL features (use normalized image)
                        mean = torch.tensor([0.485, 0.456, 0.406]).to(device).reshape(3, 1, 1)
                        std = torch.tensor([0.228, 0.224, 0.225]).to(device).reshape(3, 1, 1)
                        cnx_input = (image_tensor_float - mean) / std
                        cnx_feat = cnx_model.forward(cnx_input)[1].cpu().squeeze(0).numpy()

                    # assert the shape of the image embedding is (768,)
                    assert image_embedding.shape == (768,), f"Unexpected image embedding shape: {image_embedding.shape}"
                    # assert the shape of clip_bigG_embeddings is (256, 1664)
                    assert clip_bigG_emb.shape == (256, 1664), f"Unexpected clip_bigG shape: {clip_bigG_emb.shape}"
                    # assert the shape of vae_latents is (4, 28, 28)
                    assert vae_latent.shape == (4, 28, 28), f"Unexpected vae_latent shape: {vae_latent.shape}"
                    # assert the shape of cnx_features is (49, 512)
                    assert cnx_feat.shape == (49, 512), f"Unexpected cnx_features shape: {cnx_feat.shape}"
                    # store the image data in all_images_data
                    # (image_index / image_data / image_embedding /
                    #  clip_bigG_embeddings / vae_latents / cnx_features /
                    #  subject / split)
                    all_images_data[coco_index] = {
                        'image_index': coco_index,
                        'image_data': image_numpy,
                        'image_embedding': image_embedding,
                        'clip_bigG_embeddings': clip_bigG_emb,
                        'vae_latents': vae_latent,
                        'cnx_features': cnx_feat,
                        'subject': subject if not is_shared_image else 'shared',
                        'split': split,
                    }

    # convert to dataframe
    subjects_df = pd.DataFrame(all_subjects_data)
    images_df = pd.DataFrame(all_images_data.values())

    # save processed data to disk
    subjects_df.to_pickle(os.path.join(processed_data_dir, 'nsd_fmri_data_df.pkl'))
    images_df.to_pickle(os.path.join(processed_data_dir, 'nsd_images_df.pkl'))

if __name__ == "__main__":

    # Check if we have the ckpts for image processing
    if not os.path.exists(vae_encode_ckpt_path):
        # Download using wget
        print(f"Downloading Stable Diffusion VAE checkpoint for image processing...")
        os.system(f"wget -O {vae_encode_ckpt_path} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/sd_image_var_autoenc.pth?download=true")
    if not os.path.exists(convnext_xl_ckpt_path):
        # Download using wget
        print(f"Downloading ConvNeXt-XL checkpoint for image processing...")
        os.system(f"wget -O {convnext_xl_ckpt_path} https://huggingface.co/datasets/pscotti/mindeyev2/resolve/main/convnext_xlarge_alpha0.75_fullckpt.pth?download=true")

    # check if the processed data already exists
    subjects_df_path = os.path.join(processed_data_dir, 'nsd_fmri_data_df.pkl')
    images_df_path = os.path.join(processed_data_dir, 'nsd_images_df.pkl')
    if os.path.exists(subjects_df_path) and os.path.exists(images_df_path):
        print("Processed data already exists. Loading from disk...")
    else:
        print("Processed data not found. Processing now...")
        process_nsd_fmri_image_all_subjects()

    # read processed data
    subjects_df = pd.read_pickle(subjects_df_path)

    # filter the subjects_df to only include the wanted subjects
    print(f"Filtering the data to only include the wanted subjects: {subject_wanted}...")
    subjects_df = subjects_df[subjects_df['subject'].isin(subject_wanted)]

    # save the filtered data to disk (with subject names in the file name)
    subject_string = '_'.join(subject_wanted)
    subjects_df.to_pickle(os.path.join(processed_data_dir, f'nsd_fmri_data_df_{subject_string}.pkl'))
    print(f"Saved the filtered dataframe with shape {subjects_df.shape}")