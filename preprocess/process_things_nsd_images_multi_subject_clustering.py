# This code calculates cosine similarity between
# the image embeddings (768,) of THINGS and NSD datasets for 
# MULTIPLE NSD subjects,
# finds the best one-to-one match per subject, intersects the THINGS images
# that were successfully paired across ALL NSD subjects, then performs
# K-Means clustering on the common set.

# NOTE: Output CSV columns:
#   things_image_index, nsd_<sub1>_image_index, nsd_<sub2>_image_index, ..., cluster_label

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
import typing
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# --------------- Start of configuration --------------- #

# Paths to necessary files
processed_dir = "datasets/processed"
nsd_images_df_path = os.path.join(processed_dir, "nsd_images_df.pkl")
things_images_df_path = os.path.join(processed_dir, "things_images_df.pkl")

# The subjects we want to use
# Format: (things_subject, tuple_of_nsd_subjects)
# NOTE: we will LOOP process (not batched) — one entry per run
subjects = [('sub-01', ('sub-01', 'sub-02', 'sub-05', 'sub-07'))]

# Filter THINGS eeg test split images
filter_things_test_split = True

# top paired images we want to get for each subject
top_k = 5000
search_expansion_factor = 100

# Target image intersection size
# NOTE: how many cross subject pairings you are looking for,
#       most likely SMALLER than 'top_k'.
target_intersection_size = 1500

# number of clusters for KMeans clustering
num_of_cluster = 5

# Batch size for cosine similarity calculation
batch_size = 72

# Device configuration for torch tensors
CUDA = 0
device = torch.device(f'cuda:{CUDA}') if torch.cuda.is_available() else torch.device('cpu')

# ---------------- End of configuration ---------------- #

# Helper function to calculate cosine similarity 
def compute_cosine_similarity(things_embeddings: torch.Tensor, 
                              nsd_embeddings: torch.Tensor, 
                              batch_size: int, 
                              device: torch.device) -> torch.Tensor:
    
    """
    Compute cosine similarity matrix between THINGS and NSD embeddings in 
    batches.
    """

    # Move to device
    things_embeddings = things_embeddings.to(device)
    nsd_embeddings = nsd_embeddings.to(device)

    # Collection list for batch results
    cosine_similarity_scores = []

    # Batch-wise cosine similarity calculation
    with torch.no_grad():
        for i in tqdm(range(0, things_embeddings.shape[0], batch_size), 
                      desc = "Calculating cosine similarities"):
            
            things_batch = things_embeddings[i : i + batch_size].unsqueeze(1)  # (batch, 1, 768)
            nsd_batch = nsd_embeddings.unsqueeze(0)                            # (1, M, 768)
            batch_sim = torch.nn.functional.cosine_similarity(things_batch, nsd_batch, dim = -1)  # (batch, M)

            # Move back to CPU and store
            cosine_similarity_scores.append(batch_sim.cpu())

    return torch.cat(cosine_similarity_scores, dim = 0)  # (N, M)

# Greedy one-to-one pairing function
def greedy_one_to_one_pairing(cosine_scores: torch.Tensor, 
                              top_k: int, 
                              search_expansion_factor: int) -> typing.List[typing.Tuple[int, int, float]]:
    
    """
    Greedy one-to-one matching from a (N, M) cosine similarity matrix.
    Returns list of (things_local_idx, nsd_local_idx, cosine_similarity_score) tuples.
    """

    # Calculate the top-k items to search
    k = min(top_k * search_expansion_factor, cosine_scores.numel())

    # Get the top-k highest cosine similarity scores and their flat indices
    top_values, top_indices = torch.topk(cosine_scores.view(-1), k = k)

    paired = []
    used_things = set()
    used_nsd = set()
    M = cosine_scores.shape[1]

    # Iterate through the top indices and greedily pair if neither 
    # index has been used
    for flat_idx, score in zip(top_indices, top_values):
        t_idx = (flat_idx // M).item()
        n_idx = (flat_idx % M).item()
        if t_idx not in used_things and n_idx not in used_nsd:
            paired.append((t_idx, n_idx, score.item()))
            used_things.add(t_idx)
            used_nsd.add(n_idx)
        if len(paired) >= top_k:
            break
    return paired

###########################################################
#      Main function - start of processing pipeline       #
###########################################################
if __name__ == "__main__":

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

    # Sanity check the DataFrames
    required_columns_nsd = ['image_index', 'image_data', 'image_embedding', 'subject', 'split']
    required_columns_things = ['image_index', 'image_data', 'image_path', 'image_embedding', 'class_label', 'split']
    missing_columns_nsd = [col for col in required_columns_nsd if col not in nsd_images_df.columns]
    missing_columns_things = [col for col in required_columns_things if col not in things_images_df.columns]
    if missing_columns_nsd:
        raise ValueError(f"Missing columns in NSD DataFrame: {missing_columns_nsd}")
    if missing_columns_things:
        raise ValueError(f"Missing columns in THINGS DataFrame: {missing_columns_things}")
    print("\033[92mSuccessfully loaded NSD and THINGS images DataFrames with all required columns.\033[0m")

    # Main loop over subject configurations
    for things_subject, nsd_subjects in subjects:

        # Normalize: if a single string is passed, wrap it in a tuple
        # NOTE: this ensures backward compatibility for single
        #       THINGS-NSD subject pairing.
        if isinstance(nsd_subjects, str):
            nsd_subjects = (nsd_subjects,)

        # Sort NSD subjects for consistent naming
        nsd_subjects = tuple(sorted(nsd_subjects))

        # Create a tag of NSD subjects for file naming
        nsd_subjects_tag = "_".join(nsd_subjects)
        print(f"\n{'='*60}")
        print(f"Processing: THINGS {things_subject} -> NSD {nsd_subjects}")
        print(f"{'='*60}")

        # Load THINGS EEG data to find viewed images
        things_eeg_data_df_path = None
        for file in os.listdir(processed_dir):
            if file.startswith('things_eeg_data_df_') and file.endswith('.pkl'):
                if things_subject in file:
                    things_eeg_data_df_path = os.path.join(processed_dir, file)
                    break
        if things_eeg_data_df_path is None:
            raise FileNotFoundError(f"Could not find things_eeg_data_df for {things_subject} in {processed_dir}.")
        things_eeg_data_df: pd.DataFrame = pd.read_pickle(things_eeg_data_df_path)
        things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['subject'] == things_subject]
        if filter_things_test_split:
            things_eeg_data_df = things_eeg_data_df[things_eeg_data_df['split'] != 'test']
        things_subject_image_indices = things_eeg_data_df['image_index'].unique().tolist()
        print(f"\033[92mSuccessfully loaded THINGS {things_subject} image indices\033[0m")
        print(f"THINGS {things_subject}: {len(things_subject_image_indices)} unique images")
        del things_eeg_data_df

        # Filter THINGS images DataFrame and prepare tensors
        things_subject_images_df = things_images_df[things_images_df['image_index'].isin(things_subject_image_indices)]
        things_image_indices = torch.tensor(things_subject_images_df['image_index'].values).unsqueeze(1)      # (N, 1)
        things_image_embeddings = torch.tensor(np.stack(things_subject_images_df['image_embedding'].values))  # (N, 768)
        print(f"things_image_indices shape: {things_image_indices.shape}, embeddings shape: {things_image_embeddings.shape}")

        # Per-NSD-subject pairing
        # For each NSD subject, find one-to-one pairs with THINGS images.
        # Store: {nsd_subject: {things_img_index: (nsd_img_index, similarity_score)}}
        per_subject_pairs: typing.Dict[str, typing.Dict[int, typing.Tuple[int, float]]] = { }

        for nsd_subject in nsd_subjects:
            print(f"\n>>> Pairing with NSD {nsd_subject} >>>")

            # Load NSD subject viewed images
            # we save the 'nsd_subject_image_indices_list'
            nsd_fmri_data_df_path = None
            for file in os.listdir(processed_dir):
                if file.startswith('nsd_fmri_data_df_') and file.endswith('.pkl'):
                    if nsd_subject in file:
                        nsd_fmri_data_df_path = os.path.join(processed_dir, file)
                        break
            if nsd_fmri_data_df_path is None:
                raise FileNotFoundError(f"Could not find nsd_fmri_data_df for {nsd_subject} in {processed_dir}.")
            nsd_fmri_data_df: pd.DataFrame = pd.read_pickle(nsd_fmri_data_df_path)
            nsd_fmri_data_df = nsd_fmri_data_df[nsd_fmri_data_df['subject'] == nsd_subject]
            nsd_subject_image_indices_list = nsd_fmri_data_df['image_index'].unique().tolist()
            print(f"\033[92mSuccessfully loaded NSD {nsd_subject} image indices\033[0m")
            print(f"NSD {nsd_subject}: {len(nsd_subject_image_indices_list)} unique images")
            del nsd_fmri_data_df

            # Filter NSD images DataFrame and prepare tensors
            nsd_subject_images_df = nsd_images_df[nsd_images_df['image_index'].isin(nsd_subject_image_indices_list)]
            nsd_image_indices = torch.tensor(nsd_subject_images_df['image_index'].values).unsqueeze(1)      # (M, 1)
            nsd_image_embeddings = torch.tensor(np.stack(nsd_subject_images_df['image_embedding'].values))  # (M, 768)
            print(f"nsd_image_indices shape: {nsd_image_indices.shape}, embeddings shape: {nsd_image_embeddings.shape}")

            # Compute (or load cached) cosine similarity
            cache_suffix = "_no_things_test" if filter_things_test_split else ""
            cache_path = os.path.join(
                processed_dir,
                f"cosine_similarity_scores_subject_THINGS_{things_subject}_NSD_{nsd_subject}{cache_suffix}.pt"
            )
            if os.path.exists(cache_path):
                cosine_scores: torch.Tensor = torch.load(cache_path)
                print(f"\033[92mLoaded cached similarity ({cosine_scores.shape}) from {cache_path}\033[0m")
            else:
                cosine_scores = compute_cosine_similarity(things_image_embeddings, nsd_image_embeddings, batch_size, device)
                torch.save(cosine_scores, cache_path)
                print(f"\033[92mSaved similarity ({cosine_scores.shape}) to {cache_path}\033[0m")

            # Greedy one-to-one matching
            paired_local = greedy_one_to_one_pairing(cosine_scores, top_k, search_expansion_factor)
            if len(paired_local) < top_k:
                print(f"\033[93mWarning: Only found {len(paired_local)} pairs for NSD {nsd_subject} (requested {top_k}).\033[0m")
                # Ask user whether to continue with fewer pairs
                user_input = input(f"Continue with {len(paired_local)} pairs for NSD {nsd_subject}? (y/n): ")
                if user_input.lower() != 'y':
                    print("Terminating ...")
                    exit(0)

            # Map local indices (our extracted list idx) -> actual image indices
            # NOTE: the keys for 'subject_map' are the actual THINGS image indices, 
            #       and the values are the corresponding NSD image indices and similarity score.
            subject_map = { }
            for t_local, n_local, score in paired_local:
                t_img = things_image_indices[t_local].item()
                n_img = nsd_image_indices[n_local].item()
                subject_map[t_img] = (n_img, score)
            per_subject_pairs[nsd_subject] = subject_map
            print(f"Paired {len(subject_map)} images for NSD {nsd_subject}")

        # NOTE: at this point, 'per_subject_pairs' is a dict of dicts:
        # {
        #   'sub-01': {things_img_idx1: (nsd_img_idx1, similarity_score1), 
        #              things_img_idx2: (nsd_img_idx2, similarity_score2), ...},
        #   'sub-02': {things_img_idxA: (nsd_img_idxA, similarity_scoreA), 
        #              things_img_idxB: (nsd_img_idxB, similarity_scoreB), ...},
        #   ...
        # }

        # try to find: Intersect THINGS indices across all NSD subjects
        # NOTE: Here we find the intersection of KEYS (THINGS image index)
        common_things_indices = None
        for nsd_subject in nsd_subjects:
            # we look for the shared keys (THINGS image indices) across all subjects' pairing dicts
            subj_things_indices = set(per_subject_pairs[nsd_subject].keys())
            if common_things_indices is None:
                common_things_indices = subj_things_indices
            else:
                common_things_indices = common_things_indices.intersection(subj_things_indices)
        print(f"\nCommon THINGS images across all NSD subjects: {len(common_things_indices)}")

        # if the intersection is too small, warn the user and ask whether to continue
        if len(common_things_indices) < target_intersection_size:
            print(f"\033[93mWarning: Only {len(common_things_indices)} common THINGS images across all NSD subjects (target was {target_intersection_size}).\033[0m")
            user_input = input("Continue with this number of common images? (y/n): ")
            if user_input.lower() != 'y':
                print("Terminating ...")
                exit(0)

        # if we have more than the target intersection size, 
        # we can optionally trim it down to the top ones based on average cosine similarity
        if len(common_things_indices) > target_intersection_size:
            print(f"\033[93mTrimming common THINGS images to top {target_intersection_size} based on average similarity...\033[0m")
            avg_similarities = []
            for t_img in common_things_indices:
                sim_sum = 0
                count = 0
                for nsd_subject in nsd_subjects:
                    n_img, sim_score = per_subject_pairs[nsd_subject][t_img]
                    sim_sum += sim_score
                    count += 1
                avg_similarities.append((t_img, sim_sum / count))
            avg_similarities.sort(key = lambda x: x[1], reverse = True)
            common_things_indices = [t for t, _ in avg_similarities[:target_intersection_size]]
            # Assert no duplicates
            assert len(set(common_things_indices)) == len(common_things_indices), "Duplicate THINGS image indices found after trimming!"
            print(f"\033[92mTrimmed to {len(common_things_indices)} common THINGS images.\033[0m")

        # NOTE: at this point, 'common_things_indices' is a set of THINGS image indices
        #       that have been paired across ALL NSD subjects.
        
        # Build output DataFrame
        rows = []
        for t_img in common_things_indices:
            row = {'things_image_index': t_img}
            for nsd_subject in nsd_subjects:
                # NOTE: we don't need the similarity score here, just the paired NSD image index
                row[f'nsd_{nsd_subject}_image_index'] = per_subject_pairs[nsd_subject][t_img][0]
            rows.append(row)
        paired_images_df = pd.DataFrame(rows)
        print(f"Output CSV columns: {list(paired_images_df.columns)}")
        print(f"Example rows:\n{paired_images_df.head()}")

        # Visualize example paired images
        n_vis = min(10, len(paired_images_df))
        n_rows_vis = 1 + len(nsd_subjects)  # THINGS row + one row per NSD subject
        plt.figure(figsize = (20, 4 * n_rows_vis))
        for i in range(n_vis):
            row = paired_images_df.iloc[i]
            t_idx = int(row['things_image_index'])
            # THINGS image
            things_img = things_subject_images_df[things_subject_images_df['image_index'] == t_idx]['image_data'].values[0]
            plt.subplot(n_rows_vis, n_vis, i + 1)
            plt.imshow(things_img)
            plt.title(f"THINGS {t_idx}", fontsize = 7)
            plt.axis('off')
            # NSD images
            for s_i, nsd_subject in enumerate(nsd_subjects):
                n_idx = int(row[f'nsd_{nsd_subject}_image_index'])
                nsd_subj_images_df = nsd_images_df[nsd_images_df['image_index'] == n_idx]
                if len(nsd_subj_images_df) > 0:
                    nsd_img = nsd_subj_images_df['image_data'].values[0]
                    plt.subplot(n_rows_vis, n_vis, (s_i + 1) * n_vis + i + 1)
                    plt.imshow(nsd_img)
                    plt.title(f"NSD {nsd_subject}\n{n_idx}", fontsize = 7)
                    plt.axis('off')
        plt.suptitle(f"Example pairs: THINGS {things_subject} -> NSD {nsd_subjects}")
        plt.tight_layout()
        fig_path = os.path.join(processed_dir, f"paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subjects_tag}.png")
        plt.savefig(fig_path)
        plt.close()
        print(f"\033[92mSaved visualization to {fig_path}\033[0m")

        # KMeans clustering
        # Gather the 768-dim THINGS embeddings for the common set
        # NOTE: things_idx_to_emb with KEYS as actual THINGS image indices, 
        #       and VALUES as the corresponding embeddings.
        things_idx_to_emb: dict[int, np.ndarray] = { }
        for idx_t, emb_t in zip(things_image_indices.squeeze().tolist(), things_image_embeddings.numpy()):
            things_idx_to_emb[idx_t] = emb_t

        # Now we can get the embeddings for the common THINGS images in the intersection set
        # NOTE: shape (num paired / trimmed as specified, 768)
        paired_things_embeddings = np.stack([things_idx_to_emb[t] for t in common_things_indices])

        # Also gather NSD embeddings (mean of all subjects) for PCA visualization
        paired_nsd_embeddings = []
        for t_img in common_things_indices:
            subj_embs = []
            for nsd_subject in nsd_subjects:
                n_img_idx = per_subject_pairs[nsd_subject][t_img][0]
                n_emb = nsd_images_df[nsd_images_df['image_index'] == n_img_idx]['image_embedding'].values[0]
                subj_embs.append(n_emb)
            paired_nsd_embeddings.append(np.mean(subj_embs, axis = 0))
        paired_nsd_embeddings = np.stack(paired_nsd_embeddings)

        # Perform KMeans clustering on the THINGS embeddings of the paired set
        print(f"\n>>> Performing KMeans clustering with {num_of_cluster} clusters >>>")
        paired_all_embeddings = np.concatenate([paired_things_embeddings, paired_nsd_embeddings], axis = 0)
        kmeans = KMeans(n_clusters = num_of_cluster, random_state = 42)
        cluster_labels = kmeans.fit_predict(paired_all_embeddings)
        paired_image_labels = cluster_labels[:len(common_things_indices)]

        # Add cluster labels to the dataframe
        paired_images_df['cluster_label'] = paired_image_labels

        # Save CSV
        csv_suffix = "_no_things_test" if filter_things_test_split else ""
        out_csv_path = os.path.join(
            processed_dir,
            f"paired_images_subject_THINGS_{things_subject}_NSD_{nsd_subjects_tag}{csv_suffix}.csv"
        )
        paired_images_df.to_csv(out_csv_path, index = False)
        print(f"\033[92mSaved paired CSV to {out_csv_path}\033[0m")

        # PCA visualization with cluster labels
        pca = PCA(n_components = 3)
        things_pca = pca.fit_transform(paired_things_embeddings)
        nsd_pca = pca.fit_transform(paired_nsd_embeddings)

        fig = plt.figure(figsize = (8, 8), dpi = 300)
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(things_pca[:, 0], things_pca[:, 1], things_pca[:, 2],
                   c = paired_image_labels, cmap = 'tab10', label = 'THINGS')
        ax.scatter(nsd_pca[:, 0], nsd_pca[:, 1], nsd_pca[:, 2],
                   c = paired_image_labels, cmap = 'tab10', label = f'NSD {nsd_subjects_tag}', marker = 'x')
        for i in range(len(things_pca)):
            ax.plot([things_pca[i, 0], nsd_pca[i, 0]],
                    [things_pca[i, 1], nsd_pca[i, 1]],
                    [things_pca[i, 2], nsd_pca[i, 2]],
                    c = 'gray', alpha = 0.3)
        ax.set_title(f"PCA clusters: THINGS {things_subject} -> NSD ({nsd_subjects_tag})")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_zlabel("PCA 3")
        ax.view_init(elev = 80, azim = -45)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(processed_dir,
                                 f"paired_embeddings_pca_clusters_subject_THINGS_{things_subject}_NSD_{nsd_subjects_tag}.png"))
        plt.close()
        print(f"\033[92mSaved PCA cluster visualization.\033[0m")

        # Cluster examples - visualization
        cluster_examples: dict[int, int] = {}
        for cl in range(num_of_cluster):
            cl_indices = np.where(paired_image_labels == cl)[0]
            if len(cl_indices) > 0:
                # NOTE: we view the median example
                ex_idx = cl_indices[len(cl_indices) // 2]
                cluster_examples[cl] = ex_idx

        if cluster_examples:
            n_rows_ce = 1 + len(nsd_subjects)
            plt.figure(figsize = (20, 4 * n_rows_ce))
            for cl, ex_idx in cluster_examples.items():
                t_img_idx = common_things_indices[ex_idx]
                things_img = things_subject_images_df[things_subject_images_df['image_index'] == t_img_idx]['image_data'].values[0]
                plt.subplot(n_rows_ce, num_of_cluster, cl + 1)
                plt.imshow(things_img)
                plt.title(f"CLUSTER({cl}) THINGS {t_img_idx}", fontsize = 7)
                plt.axis('off')
                for s_i, nsd_subject in enumerate(nsd_subjects):
                    n_img_idx = per_subject_pairs[nsd_subject][t_img_idx][0]
                    nsd_subj_imgs = nsd_images_df[nsd_images_df['image_index'] == n_img_idx]
                    if len(nsd_subj_imgs) > 0:
                        plt.subplot(n_rows_ce, num_of_cluster, (s_i + 1) * num_of_cluster + cl + 1)
                        plt.imshow(nsd_subj_imgs['image_data'].values[0])
                        plt.title(f"CLUSTER({cl}) NSD {nsd_subject}\n{n_img_idx}", fontsize = 7)
                        plt.axis('off')
            plt.suptitle(f"Cluster examples: THINGS {things_subject} -> NSD ({nsd_subjects_tag})")
            plt.tight_layout()
            plt.savefig(os.path.join(processed_dir,
                                     f"cluster_examples_subject_THINGS_{things_subject}_NSD_{nsd_subjects_tag}.png"))
            plt.close()
            print(f"\033[92mSaved cluster example visualization.\033[0m")

        print(f"\nDone processing THINGS {things_subject} -> NSD {nsd_subjects}.")