# This script is used to repack the EEG data from the THINGS dataset.
# we use CLIPModel from transformers to process the
# embedding of GT images.
# transformer package is required to run this script.
# run "pip install transformers" to install the package.

# We'll save the processed data in pandas.DataFrame (eeg / class_label / image_index / subject / split).
# We'll also index every unique images in 'img_set_path' and save the
# index in a pandas.DataFrame for later use. (image_index / image_data / image_path / image_embedding / class_label / split)

# WARNING: 
#     - training: num of trails =  4 per image
#     - testing:  num of trails = 80 per image

# --------------- Start of configuration --------------- #

# import os
import os
# set transformers cache directory
os.environ['TRANSFORMERS_CACHE'] = 'datasets/transformers_cache'
os.environ['HF_HOME'] = 'datasets/transformers_cache'
os.environ['HF_HUB_CACHE'] = 'datasets/transformers_cache'
# Set Hugging Face mirror (if needed)
# os.environ["HF_ENDPOINT"] = "<MIRROR END POINT>"
# set proxy (if needed)
# os.environ['NO_PROXY'] = 'huggingface.co'

# import necessary libraries
import torch
import pandas as pd
import numpy as np
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from tqdm import tqdm

# Data path (root folder to all subjects / images)
data_path = 'datasets/THINGS-EEG/Preprocessed_data_250Hz_whiten'
img_set_path = 'datasets/THINGS-EEG/Image_set_Resize' # (224, 224) colour images
processed_data_dir = 'datasets/processed'
os.makedirs(processed_data_dir, exist_ok = True)

# we specify the subjects we want to process
subject_wanted = ['sub-01'] # 10 in total

# CUDA device configuration
CUDA = 0

# ---------------- End of configuration ---------------- #

def process_things_image_data() -> None:
    # We first load all images
    # Get all image file names in the img_set_path
    # There are two folders under img_set_path: test_images and train_images
    # inside test_images or train_images, there are sub folders containing the images.
    # the sub folder is named <class_id>_<class name>
    # inside the sub folder, the images are named <class name>_<random idx>.jpg
    # We load training and testing in separate lists
    train_images = []
    test_images = []
    print(f"Loading images from {img_set_path}...")
    for split in ['train_images', 'test_images']:
        split_path = os.path.join(img_set_path, split)
        for class_folder in os.listdir(split_path):
            class_folder_path = os.path.join(split_path, class_folder)
            if os.path.isdir(class_folder_path):
                for img_file in os.listdir(class_folder_path):
                    img_file_path = os.path.join(class_folder_path, img_file)
                    if split == 'train_images':
                        train_images.append(img_file_path)
                    else:
                        test_images.append(img_file_path)
    # Show counts and first few of the lists
    print(f"Number of training images: {len(train_images)}")
    print(f"Number of testing images: {len(test_images)}")
    # Number of training images: 16540
    # Number of testing images: 200
    # NOTE: paths are stored in relative to project directory
    #       i.e. 'datasets/THINGS-EEG/Image_set_Resize/train_images/00001_aardvark/aardvark_01b.jpg'

    # we sort the image paths to ensure the order is consistent
    train_images.sort()
    test_images.sort()

    # we create dataframes for the images
    train_images_df = pd.DataFrame(train_images, columns = ['image_path'])
    test_images_df = pd.DataFrame(test_images, columns = ['image_path'])

    # we add split column
    train_images_df['split'] = 'train'
    test_images_df['split'] = 'test'

    # we concatenate the dataframes
    images_df = pd.concat([train_images_df, test_images_df], ignore_index = True)

    # we generate the index for the images
    images_df['image_index'] = images_df.index

    # we extract the class label from the image path
    # the class label is the name of the sub folder, which is the second last part of the path
    # note that we have some classes with name '00001_aircraft_carrier'
    # we shall handle double '_' right
    images_df['class_label'] = images_df['image_path'].apply(lambda x: x.split('/')[-2].split('_', 1)[1])

    # We load the images it self to the dataframe
    def load_image(image_path) -> np.ndarray:
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            return image_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    # we apply the function to the dataframe
    print("Loading images into dataframe...")
    tqdm.pandas(desc = "Loading images")
    images_df['image_data'] = images_df['image_path'].progress_apply(load_image)
    # we check if there are any images that failed to load
    failed_images = images_df[images_df['image_data'].isnull()]
    if len(failed_images) > 0:
        print(f"Failed to load {len(failed_images)} images:")
        print(failed_images['image_path'])
    else:
        print("All images loaded successfully.")

    # we use CLIPModel to process the images and get the embeddings
    # our image already in the format of (224, 224, 3)
    # we directly use get_image_features() to get the embeddings
    device = torch.device(f'cuda:{CUDA}') if torch.cuda.is_available() else torch.device('cpu')
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    def get_image_embedding(image_data: np.ndarray) -> np.ndarray:
        try:
            inputs = processor(images = image_data, return_tensors = "pt").to(device)
            with torch.no_grad():
                # Get the actual embeddings from the output dictionary
                outputs = model.get_image_features(**inputs)
                image_embedding = outputs.pooler_output.cpu().numpy().squeeze()
                # assert shape is (768,)
                assert image_embedding.shape == (768,), f"Expected shape (768,), got {image_embedding.shape}"
            return image_embedding
        except Exception as e:
            raise ValueError(f"Error processing image for embedding: {e}")
    # we apply the function to the dataframe
    print("Processing images to get embeddings...")
    tqdm.pandas(desc = "Processing image embeddings")
    images_df['image_embedding'] = images_df['image_data'].progress_apply(get_image_embedding)
    # we check if there are any images that failed to process
    failed_embeddings = images_df[images_df['image_embedding'].isnull()]
    if len(failed_embeddings) > 0:
        print(f"Failed to process {len(failed_embeddings)} images for embedding:")
        print(failed_embeddings['image_path'])
    else:
        print("All images processed successfully for embedding.")

    # we save the images dataframe to pickle
    images_df.to_pickle(os.path.join(processed_data_dir, 'things_images_df.pkl'))

def process_things_eeg_data(subject_wanted: list) -> None:
    # we then process the EEG data and save to pickle
    # we load the data for each subject under data_path (search all directories)
    # we save the data in a dataframe with columns: (eeg / class_label / image_index / subject / split)
    subjects = []
    for subject_folder in os.listdir(data_path):
        subject_folder_path = os.path.join(data_path, subject_folder)
        if os.path.isdir(subject_folder_path):
            subjects.append(subject_folder)
    print(f"Found subjects: {subjects}")

    # we only keep subjects we want
    subjects = [subject for subject in subjects if subject in subject_wanted]
    print(f"Processing subjects: {subjects}")

    # create an empty dataframe to store the data
    eeg_data_df = pd.DataFrame(columns = ['eeg', 'class_label', 'image_index', 'subject', 'split'])

    # we define a function to process the data for one subject and one split (train/test)
    def process_data(data: dict, subject: str, split: str, image_df: pd.DataFrame) -> pd.DataFrame:
            eeg = data['eeg']  # shape: (16540, 4, 63, 250) for train, (200, 80, 63, 250) for test
            img_path: np.ndarray[str] = data['img']  # shape: (16540, 4) for train, (200, 80) for test

            # we loop through the samples and create a dataframe
            # 4 recordings stored together
            data_list = []
            for i in tqdm(range(eeg.shape[0]), desc = f"Processing {subject} {split} data"):
                # slice the eeg data for the sample
                sample_eeg = eeg[ i , : ]  # shape: (4, 63, 250)
                # get the image path for the sample
                sample_img_path: str = img_path[ i , 0 ]
                # we try to look up this image path in image_df
                # note that the two paths have different relative roots,
                # but the endings should be the same, so we can match them by the ending of the path
                matched_image = image_df[image_df['image_path'].apply(lambda x: x.endswith(sample_img_path))]
                if len(matched_image) == 0:
                    raise ValueError(f"No matching image found for sample {i} with image path {sample_img_path}")
                elif len(matched_image) > 1:
                    raise ValueError(f"Multiple matching images found for sample {i} with image path {sample_img_path}")
                else:
                    matched_image = matched_image.iloc[0]
                    class_label = matched_image['class_label']
                    image_index = matched_image['image_index']
                data_list.append({
                    'eeg': sample_eeg,
                    'class_label': class_label,
                    'image_index': image_index,
                    'subject': subject,
                    'split': split
                })
            return pd.DataFrame(data_list)

    # we load the image dataframe
    image_df = pd.read_pickle(os.path.join(processed_data_dir, 'things_images_df.pkl'))
    print(f"Loaded image dataframe with shape {image_df.shape}")

    # we loop through the subjects and load the data
    for subject in subjects:
        print(f"Processing subject {subject}...")
        # Get the folder path for the subject
        subject_folder_path = os.path.join(data_path, subject)
        # we load the train and test data
        # under each subject folder, there are two files: train.pt and test.pt
        train_data_path = os.path.join(subject_folder_path, 'train.pt')
        test_data_path = os.path.join(subject_folder_path, 'test.pt')
        if os.path.exists(train_data_path):
            train_data = torch.load(train_data_path, weights_only = False)
            print(f"Loaded train data for {subject}")
        else:
            print(f"Train data not found for {subject}, skipping...")
            continue
        if os.path.exists(test_data_path):
            test_data = torch.load(test_data_path, weights_only = False)
            print(f"Loaded test data for {subject}")
        else:
            print(f"Test data not found for {subject}, skipping...")
            continue

        # we process the train and test data separately
        # we extract the eeg, label, img, text, session from the data
        # we create a dataframe for the train and test data with columns: (eeg / class_label / image_index / subject / split)
        # we process the train and test data
        train_df = process_data(train_data, subject, 'train', image_df)
        test_df = process_data(test_data, subject, 'test', image_df)
        # we concatenate the train and test dataframes
        eeg_data_df = pd.concat([eeg_data_df, train_df, test_df], ignore_index = True)
    
    # we save the combined dataframe
    # generate file name with the subjects we processed
    subject_str = '_'.join(subjects)
    eeg_data_df.to_pickle(os.path.join(processed_data_dir, f'things_eeg_data_df_{subject_str}.pkl'))
    print(f"Saved combined EEG data dataframe with shape {eeg_data_df.shape}")

if __name__ == "__main__":
    # if the processed data already exists, we skip processing
    # if not, we process
    if os.path.exists(os.path.join(processed_data_dir, 'things_images_df.pkl')):
        print("Processed image data already exists. Skipping processing.")
    else:
        process_things_image_data()

    # we process the EEG data for the specified subjects
    process_things_eeg_data(subject_wanted)