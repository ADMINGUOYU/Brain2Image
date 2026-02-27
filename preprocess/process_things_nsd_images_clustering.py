# This code would calculate cosine similarity between
# the image embeddings (768,) of THINGS and NSD datasets,
# We choose the n most similar images between the two datasets
# and pair them.
# We then try to re-label the paired images them by PCA analysis
# and performing clustering.

# NOTE: We will output a csv file containing the paired image indexes
# and their cluster labels.

# NOTE: We load image datasets from (make sure to run the repacking
#       code under ./datasets first):
# - datasets/processed/nsd_images_df.pkl
#   -> pandas.DataFrame with columns: 
#      (image_index / image_data / image_embedding / subject / split)
# - datasets/processed/things_images_df.pkl
#   -> pandas.DataFrame with columns:
#      (image_index / image_data / image_path / image_embedding / class_label / split)

# Imports
import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --------------- Start of configuration --------------- #

# Paths to necessary files
processed_dir = "datasets/processed"
nsd_images_df_path = os.path.join(processed_dir, "nsd_images_df.pkl")
things_images_df_path = os.path.join(processed_dir, "things_images_df.pkl")

# The subjects we want to use (only process the image viewed by these subjects)
# NOTE: we will LOOP process (not batched)
# NOTE: in (things subject, nsd subject)
subjects =  [('sub-01', 'sub-01')]

# Filter THINGS eeg test split images
# NOTE: THINGS test set has 80 trails instead of 4 per image
# NOTE: we append "_no_things_test" to the output file name if we filter the test split to avoid confusion
filter_things_test_split = False

# top paired images we want to get for each subject
# NOTE: if we don't have that amount of paired images,
#       we get as much as we can
#       A warning will be raised in that case.
top_k = 1000
search_expansion_factor = 20 # we will get top_k * search_expansion_factor pairs first and then filter to ensure one-to-one pairing

# number of clusters for KMeans clustering
num_of_cluster = 5

# Batch size for cosine similarity calculation (to avoid Out-of-Memory issues)
# NOTE: we will calculate cosine similarity in batches to avoid OOM,
#       since the number of images can be large (e.g., 73k in NSD).
#       We can adjust this batch size based on the available GPU memory.
batch_size = 72

# Device configuration for torch tensors
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------- End of configuration ---------------- #

# Make sure the files exist
if not os.path.exists(nsd_images_df_path):
    raise FileNotFoundError(f"NSD images DataFrame file not found: {nsd_images_df_path}, \
                            please make sure to run the repacking code under ./datasets first.")
if not os.path.exists(things_images_df_path):
    raise FileNotFoundError(f"THINGS images DataFrame file not found: {things_images_df_path}, \
                            please make sure to run the repacking code under ./datasets first.")

# Load the DataFrames
nsd_images_df: pd.DataFrame = pd.read_pickle(nsd_images_df_path)
things_images_df: pd.DataFrame = pd.read_pickle(things_images_df_path)

# Sanity check the DataFrames (make sure we have all necessary columns)
required_columns_nsd = ['image_index', 'image_data', 'image_embedding', 'subject', 'split']
required_columns_things = ['image_index', 'image_data', 'image_path', 'image_embedding', 'class_label', 'split']
missing_columns_nsd = [col for col in required_columns_nsd if col not in nsd_images_df.columns]
missing_columns_things = [col for col in required_columns_things if col not in things_images_df.columns]
if missing_columns_nsd:
    raise ValueError(f"Missing columns in NSD DataFrame: {missing_columns_nsd}")
if missing_columns_things:
    raise ValueError(f"Missing columns in THINGS DataFrame: {missing_columns_things}")
print("Successfully loaded NSD and THINGS images DataFrames with all required columns.")

# we loop every subject we want
for things_subject, nsd_subject in subjects:
    # verbose
    print(f"Processing subject pair: THINGS {things_subject} NSD {nsd_subject}...")

    # We need to get the images viewed (THINGS) from the processed EEG dataset
    # We first locate all 'things_eeg_data_df_XXX.pkl' and find one which XXX contains the current subject
    things_eeg_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('things_eeg_data_df_') and file.endswith('.pkl'):
            if things_subject in file:
                things_eeg_data_df_path = os.path.join(processed_dir, file)
                break
    if things_eeg_data_df_path is None:
        raise FileNotFoundError(f"Could not find the things_eeg_data_df file for subject {things_subject} in {processed_dir}.")
    # Load the things_eeg_data_df containing the current subject
    # dataframe columns: (eeg / class_label / image_index / subject / split)
    things_eeg_data_df = pd.read_pickle(things_eeg_data_df_path)
    # we only filter the rows which subject == current subject
    things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
    # verbose
    print(f"Number of rows in things_eeg_data_df for {things_subject}: {len(things_eeg_data_df)}")
    # Drop test split if specified
    if filter_things_test_split:
        things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['split'] != 'test']
        print(f"Number of rows in things_eeg_data_df for {things_subject} after filtering test split: {len(things_eeg_data_df)}")
    # we get the unique image indices viewed by the current subject
    # save it to list and unload the things_eeg_data_df to save memory
    things_subject_image_indices = things_eeg_data_df['image_index'].unique().tolist()
    print(f"Number of unique images viewed by {things_subject} in THINGS: {len(things_subject_image_indices)}")
    del things_eeg_data_df

    # we also open the nsd_fmri_data_df
    nsd_fmri_data_df_path = None
    for file in os.listdir(processed_dir):
        if file.startswith('nsd_fmri_data_df_') and file.endswith('.pkl'):
            if nsd_subject in file:
                nsd_fmri_data_df_path = os.path.join(processed_dir, file)
                break
    if nsd_fmri_data_df_path is None:
        raise FileNotFoundError(f"Could not find the nsd_fmri_data_df file for subject {nsd_subject} in {processed_dir}.")
    # Load the nsd_fmri_data_df containing the current subject
    # dataframe columns: (fmri / image_index / subject / split / sample_id / tar_path)
    nsd_fmri_data_df = pd.read_pickle(nsd_fmri_data_df_path)
    # we only filter the rows which subject == current subject
    nsd_fmri_data_df = nsd_fmri_data_df[nsd_fmri_data_df['subject'] == nsd_subject]
    # verbose
    print(f"Number of rows in nsd_fmri_data_df for {nsd_subject}: {len(nsd_fmri_data_df)}")
    # we get the unique image indices viewed by the current subject
    # save it to list and unload the nsd_fmri_data_df to save memory
    nsd_subject_image_indices = nsd_fmri_data_df['image_index'].unique().tolist()
    print(f"Number of unique images viewed by {nsd_subject} in NSD: {len(nsd_subject_image_indices)}")
    del nsd_fmri_data_df

    # we filter the things_images_df to only include the images viewed by the current subject
    things_subject_images_df = things_images_df[things_images_df['image_index'].isin(things_subject_image_indices)]
    print(f"Number of rows in things_subject_images_df for {things_subject}: {len(things_subject_images_df)}")
    # we also filter the nsd_images_df to only include the images viewed by the current subject
    nsd_subject_images_df = nsd_images_df[nsd_images_df['image_index'].isin(nsd_subject_image_indices)]
    print(f"Number of rows in nsd_subject_images_df for {nsd_subject}: {len(nsd_subject_images_df)}")

    # we convert the image index column and image embedding column to torch tensor 
    # for later cosine similarity calculation and stack them into 
    # (N, 1) and (N, 768) respectively, we don't need the dataframe for now
    things_image_indices = torch.tensor(things_subject_images_df['image_index'].values).unsqueeze(1) # (N, 1)
    things_image_embeddings = torch.tensor(np.stack(things_subject_images_df['image_embedding'].values)) # (N, 768)
    nsd_image_indices = torch.tensor(nsd_subject_images_df['image_index'].values).unsqueeze(1) # (M, 1)
    nsd_image_embeddings = torch.tensor(np.stack(nsd_subject_images_df['image_embedding'].values)) # (M, 768)
    print(f"things_image_indices shape: {things_image_indices.shape}, things_image_embeddings shape: {things_image_embeddings.shape}")
    print(f"nsd_image_indices shape: {nsd_image_indices.shape}, nsd_image_embeddings shape: {nsd_image_embeddings.shape}")

    # check if we have the cached similarity matrix file ready
    cosine_similarity_cache_path = os.path.join(processed_dir, f"cosine_similarity_scores_subject_THINGS_{things_subject}_NSD_{nsd_subject}.pt")
    # If drop test append "_no_things_test" to the cache file name to avoid confusion
    if filter_things_test_split:
        cosine_similarity_cache_path = os.path.join(processed_dir, f"cosine_similarity_scores_subject_THINGS_{things_subject}_NSD_{nsd_subject}_no_things_test.pt")
    # Load cache or compute similarity scores
    if os.path.exists(cosine_similarity_cache_path):
        cosine_similarity_scores: torch.Tensor = torch.load(cosine_similarity_cache_path)
        print(f"Loaded cached cosine similarity scores for subject THINGS {things_subject} and NSD {nsd_subject} from {cosine_similarity_cache_path}.")
        print(f"Cosine similarity scores shape: {cosine_similarity_scores.shape}")
        print(f"If this is unexpected, please delete the cache file at {cosine_similarity_cache_path} and re-run the code to recalculate the similarity scores.")
    else:
        # we calculate the cosine similarity between every pairs
        # (should have N x M similarity scores)
        print("Calculating cosine similarity between THINGS and NSD image embeddings...")
        # we can use torch.nn.functional.cosine_similarity, but it only calculates the similarity between
        # two tensors of the same shape, so we need to expand the dimensions of the tensors
        # things_image_embeddings: (N, 768) -> (N, 1, 768)
        # nsd_image_embeddings: (M, 768) -> (1, M, 768)
        # then we can calculate the cosine similarity along the last dimension and get a (N, M) tensor of similarity scores
        cosine_similarity_scores = []
        with torch.no_grad():
            for i in tqdm(range(0, things_image_embeddings.shape[0], batch_size), desc = "Calculating cosine similarities"):
                things_batch = things_image_embeddings[ i : i + batch_size ].unsqueeze(1) # (batch_size, 1, 768)
                nsd_batch = nsd_image_embeddings.unsqueeze(0) # (1, M, 768)
                batch_similarity = torch.nn.functional.cosine_similarity(things_batch, nsd_batch, dim = -1) # (batch_size, M)
                cosine_similarity_scores.append(batch_similarity.cpu())
        cosine_similarity_scores = torch.cat(cosine_similarity_scores, dim = 0) # (N, M)
        print(f"Cosine similarity scores shape: {cosine_similarity_scores.shape}")

        # cache the cosine similarity score matrix
        torch.save(cosine_similarity_scores, os.path.join(processed_dir, f"cosine_similarity_scores_subject_THINGS_{things_subject}_NSD_{nsd_subject}.pt"))
        print(f"Saved cosine similarity scores for subject THINGS {things_subject} and NSD {nsd_subject} to {os.path.join(processed_dir, f'cosine_similarity_scores_subject_THINGS_{things_subject}_NSD_{nsd_subject}.pt')}")

    # we then get the top K paired (make sure images are one-to-one -> we don't
    # want one THINGS eeg data to be paired with multiple NSS fmri data)
    # if so, we only keep the closest pair
    print(f"Getting top {top_k} paired images between THINGS and NSD...")
    # we get the score ranking across the (M, N) matrix and get the top k pairs
    # we get all first and then filter to make sure one-to-one pairing
    top_values, top_indices = torch.topk(cosine_similarity_scores.view(-1), k = min(top_k * search_expansion_factor, cosine_similarity_scores.numel()))
    paired_indices = []
    paired_things_indices = set()
    paired_nsd_indices = set()
    print("Starting to filter top pairs to ensure one-to-one pairing...")
    for idx, top_idx in enumerate(top_indices):
        things_idx = top_idx // nsd_image_embeddings.shape[0] # get the THINGS image index
        nsd_idx = top_idx % nsd_image_embeddings.shape[0] # get the NSD image index
        if things_idx.item() not in paired_things_indices and nsd_idx.item() not in paired_nsd_indices:
            paired_indices.append((things_idx.item(), nsd_idx.item())) # we add this pair of indices to the list
            paired_things_indices.add(things_idx.item()) # mark this THINGS index as paired
            paired_nsd_indices.add(nsd_idx.item()) # mark this NSD index as paired
        if len(paired_indices) >= top_k:
            break
    if len(paired_indices) < top_k:
        print(f"Warning: Only found {len(paired_indices)} paired images for subject THINGS {things_subject} and NSD {nsd_subject}, which is less than the requested top_k = {top_k}.\n\
              Consider increasing the search_expansion_factor or check the cosine similarity scores to see if there are enough similar images between the two datasets.")

    # We map back to image indexes
    paired_image_indices = []
    for things_idx, nsd_idx in paired_indices:
        things_image_index = things_image_indices[things_idx].item() # get the original image index for this THINGS image
        nsd_image_index = nsd_image_indices[nsd_idx].item() # get the original image index for this NSD image
        paired_image_indices.append((things_image_index, nsd_image_index)) # we add this pair of image indices to the list
    print(f"Example paired image indices (THINGS index, NSD index): {paired_image_indices[:5]}")
    # get the images data for the paired images and visualize some of them
    things_image_data = []
    nsd_image_data = []
    for i in range(min(10, len(paired_image_indices))):
        things_image_index, nsd_image_index = paired_image_indices[i]
        things_image_data.append(things_subject_images_df[things_subject_images_df['image_index'] == things_image_index]['image_data'].values[0])
        nsd_image_data.append(nsd_subject_images_df[nsd_subject_images_df['image_index'] == nsd_image_index]['image_data'].values[0])
    # visualize the paired images
    plt.figure(figsize = (20, 4))
    for i in range(len(things_image_data)):
        plt.subplot(2, len(things_image_data), i + 1)
        plt.imshow(things_image_data[i])
        plt.title(f"THINGS idx: {paired_image_indices[i][0]}")
        plt.axis('off')
        plt.subplot(2, len(nsd_image_data), len(things_image_data) + i + 1)
        plt.imshow(nsd_image_data[i])
        plt.title(f"NSD idx: {paired_image_indices[i][1]}")
        plt.axis('off')
    plt.suptitle(f"Example paired images for THINGS subject {things_subject} and NSD subject {nsd_subject}")
    plt.tight_layout()
    # save the figure
    plt.savefig(os.path.join(processed_dir, f"paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png"))
    # del the image data to save memory
    del things_image_data
    del nsd_image_data

    # save the index of the paired images to a csv file for later use
    paired_images_df = pd.DataFrame(paired_image_indices, columns = ['things_image_index', 'nsd_image_index'])
    # If filtered test split appen "_no_things_test" to csv file name
    out_csv_path = os.path.join(processed_dir, f"paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subject}.csv")
    if filter_things_test_split:
        out_csv_path = os.path.join(processed_dir, f"paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subject}_no_things_test.csv")
    paired_images_df.to_csv(out_csv_path, index = False)
    print(f"Saved paired image indices for subject THINGS {things_subject} and NSD {nsd_subject} to {out_csv_path}")

    # Now we want to do PCA analysis to the paired images perform clustering
    # fetch the embeddings upon the indexes we got
    paired_things_embeddings = []
    paired_nsd_embeddings = []
    for things_idx, nsd_idx in paired_indices:
        paired_things_embeddings.append(things_image_embeddings[things_idx].cpu().numpy())
        paired_nsd_embeddings.append(nsd_image_embeddings[nsd_idx].cpu().numpy())
    paired_things_embeddings = np.stack(paired_things_embeddings) # (num_pairs, 768)
    paired_nsd_embeddings = np.stack(paired_nsd_embeddings) # (num_pairs, 768)
    print(f"Paired THINGS embeddings shape: {paired_things_embeddings.shape}, Paired NSD embeddings shape: {paired_nsd_embeddings.shape}")
    # we can then perform PCA on the paired embeddings and visualize the first 3 principal components
    print("Performing PCA on paired embeddings...")
    pca = PCA(n_components = 3)
    paired_things_embeddings_pca = pca.fit_transform(paired_things_embeddings) # (num_pairs, 3)
    paired_nsd_embeddings_pca = pca.fit_transform(paired_nsd_embeddings) # (num_pairs, 3)
    print(f"Paired THINGS embeddings PCA shape: {paired_things_embeddings_pca.shape}, Paired NSD embeddings PCA shape: {paired_nsd_embeddings_pca.shape}")
    # visualize the PCA results (pair-wise version) (3D plot)
    fig = plt.figure(figsize = (8, 8), dpi = 300)
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(paired_things_embeddings_pca[:, 0], paired_things_embeddings_pca[:, 1], paired_things_embeddings_pca[:, 2], c = 'aquamarine', label = 'THINGS')
    ax.scatter(paired_nsd_embeddings_pca[:, 0], paired_nsd_embeddings_pca[:, 1], paired_nsd_embeddings_pca[:, 2], c = 'mediumpurple', label = 'NSD')
    for i in range(paired_things_embeddings_pca.shape[0]):
        ax.plot([paired_things_embeddings_pca[i, 0], paired_nsd_embeddings_pca[i, 0]], 
                 [paired_things_embeddings_pca[i, 1], paired_nsd_embeddings_pca[i, 1]], 
                 [paired_things_embeddings_pca[i, 2], paired_nsd_embeddings_pca[i, 2]],
                 c = 'gray', alpha = 0.5)
    ax.set_title(f"PCA for THINGS subject {things_subject} and NSD subject {nsd_subject} (Pair-wise)")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    # select best view angle
    ax.view_init(elev = 80, azim = -45)
    # show legend and make it look better
    ax.legend()
    fig.tight_layout()
    # save the figure
    fig.savefig(os.path.join(processed_dir, f"paired_embeddings_pca_pairwise_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png"))
    print(f"Saved pair-wise PCA visualization of paired embeddings for subject THINGS {things_subject} and NSD {nsd_subject} to {os.path.join(processed_dir, f'paired_embeddings_pca_pairwise_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png')}")

    # then we give labels to these clusters
    # we can use KMeans clustering to cluster the paired embeddings into k clusters
    print(f"Performing KMeans clustering with {num_of_cluster} clusters on paired embeddings...")
    kmeans = KMeans(n_clusters = num_of_cluster, random_state = 42)
    paired_embeddings = np.concatenate([paired_things_embeddings, paired_nsd_embeddings], axis = 0) # (2 * num_pairs, 768)
    cluster_labels = kmeans.fit_predict(paired_embeddings) # (2 * num_pairs,)
    print(f"Cluster labels shape: {cluster_labels.shape}, Cluster centers shape: {kmeans.cluster_centers_.shape}")
    # we assign the cluster label to each paired image
    # add to paired_images_df and overwrite the csv file
    paired_image_labels = cluster_labels[:len(paired_indices)] # (num_pairs,)
    paired_images_df['cluster_label'] = paired_image_labels
    paired_images_df.to_csv(out_csv_path, index = False)
    print(f"Saved paired image indices with cluster labels for subject THINGS {things_subject} and NSD {nsd_subject} to {os.path.join(processed_dir, f'paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subject}.csv')}")
    # visualize the clusters in the PCA space (3D plot)
    fig = plt.figure(figsize = (8, 8), dpi = 300)
    ax = fig.add_subplot(111, projection = '3d')
    scatter = ax.scatter(paired_things_embeddings_pca[:, 0], paired_things_embeddings_pca[:, 1], paired_things_embeddings_pca[:, 2], c = paired_image_labels, cmap = 'tab10', label = 'THINGS')
    scatter = ax.scatter(paired_nsd_embeddings_pca[:, 0], paired_nsd_embeddings_pca[:, 1], paired_nsd_embeddings_pca[:, 2], c = paired_image_labels, cmap = 'tab10', label = 'NSD', marker = 'x')
    for i in range(paired_things_embeddings_pca.shape[0]):
        ax.plot([paired_things_embeddings_pca[i, 0], paired_nsd_embeddings_pca[i, 0]], 
                 [paired_things_embeddings_pca[i, 1], paired_nsd_embeddings_pca[i, 1]], 
                 [paired_things_embeddings_pca[i, 2], paired_nsd_embeddings_pca[i, 2]],
                 c = 'gray', alpha = 0.5)
    ax.set_title(f"PCA for THINGS subject {things_subject} and NSD subject {nsd_subject} with Cluster Labels")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")
    # select best view angle
    ax.view_init(elev = 80, azim = -45)
    # show legend and make it look better
    ax.legend()
    fig.tight_layout()
    # save the figure
    fig.savefig(os.path.join(processed_dir, f"paired_embeddings_pca_clusters_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png"))
    print(f"Saved PCA visualization of paired embeddings with cluster labels for subject THINGS {things_subject} and NSD {nsd_subject} to {os.path.join(processed_dir, f'paired_embeddings_pca_clusters_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png')}")

    # fetch 1 pair from each cluster and visualize them
    cluster_examples = {}
    for cluster_label in range(num_of_cluster):
        cluster_indices = np.where(paired_image_labels == cluster_label)[0]
        if len(cluster_indices) > 0:
            example_idx = cluster_indices[len(cluster_indices) // 2]
            # load the image data
            things_image_index = paired_image_indices[example_idx][0]
            nsd_image_index = paired_image_indices[example_idx][1]
            things_image_data = things_subject_images_df[things_subject_images_df['image_index'] == things_image_index]['image_data'].values[0]
            nsd_image_data = nsd_subject_images_df[nsd_subject_images_df['image_index'] == nsd_image_index]['image_data'].values[0]
            cluster_examples[cluster_label] = (things_image_index, nsd_image_index, things_image_data, nsd_image_data)
    # visualize the cluster examples
    plt.figure(figsize = (20, 4))
    for cluster_label, (things_image_index, nsd_image_index, things_image_data, nsd_image_data) in cluster_examples.items():
        plt.subplot(2, num_of_cluster, cluster_label + 1)
        plt.imshow(things_image_data)
        plt.title(f"Cluster {cluster_label} - THINGS idx: {things_image_index}")
        plt.axis('off')
        plt.subplot(2, num_of_cluster, num_of_cluster + cluster_label + 1)
        plt.imshow(nsd_image_data)
        plt.title(f"Cluster {cluster_label} - NSD idx: {nsd_image_index}")
        plt.axis('off')
    plt.suptitle(f"Example paired images from each cluster for THINGS subject {things_subject} and NSD subject {nsd_subject}")
    plt.tight_layout()
    # save the figure
    plt.savefig(os.path.join(processed_dir, f"cluster_examples_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png"))
    print(f"Saved example paired images from each cluster for subject THINGS {things_subject} and NSD {nsd_subject} to {os.path.join(processed_dir, f'cluster_examples_subject_THINGS_{things_subject}_NSD_{nsd_subject}.png')}")
    # del the image data to save memory
    del cluster_examples