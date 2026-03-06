# This script will convert e2e checkpoint to alignment checkpoint
# Meant to be used after EEG ONLY e2e training


# --------------- Start of configuration --------------- #

# Path to the e2e checkpoint to be converted
e2e_checkpoint_path = '...'

# Optional: Path to save the converted alignment checkpoint 
# NOTE: (if not provided, it will be saved in the same directory with '_align' suffix)
align_checkpoint_path = None

# ---------------- End of configuration ---------------- #


# Import necessary libraries
import os
import torch

# Function to convert e2e checkpoint to alignment checkpoint
def convert_e2e_to_align(e2e_checkpoint_path: str, 
                         align_checkpoint_path: str = None,
                         model_key: str = 'align_model',
                         parameters_key: str = 'parameters'):
    
    # Load the e2e checkpoint
    e2e_checkpoint = torch.load(e2e_checkpoint_path, map_location = 'cpu')

    # Extract the model state dict from the e2e checkpoint
    if model_key not in e2e_checkpoint:
        raise KeyError(f"'{model_key}' not found in the e2e checkpoint.")
    if parameters_key not in e2e_checkpoint:
        raise KeyError(f"'{parameters_key}' not found in the e2e checkpoint.")
    
    model_state_dict = e2e_checkpoint[model_key]
    parameters = e2e_checkpoint[parameters_key]

    # Create a new checkpoint for alignment training
    # Target keys: 
    # dict_keys(['eeg_encoder', 'projection_head', 'parameters'])
    align_checkpoint = {
        'eeg_encoder': model_state_dict.get('eeg_encoder', {}),
        'projection_head': model_state_dict.get('projection_head', {}),
        'parameters': parameters
    }

    # Save the new checkpoint
    if align_checkpoint_path is None:
        base_name, ext = os.path.splitext(e2e_checkpoint_path)
        align_checkpoint_path = base_name + '_align' + ext
    torch.save(align_checkpoint, align_checkpoint_path)
    
    # verbose
    print(f"Alignment checkpoint saved to: {align_checkpoint_path}")

# Main function to execute the conversion
if __name__ == "__main__":

    # Check if the path exists
    if not os.path.exists(e2e_checkpoint_path):
        raise FileNotFoundError(f"The specified e2e checkpoint path does not exist: {e2e_checkpoint_path}")
    
    # CALL the conversion function
    convert_e2e_to_align(e2e_checkpoint_path, align_checkpoint_path)