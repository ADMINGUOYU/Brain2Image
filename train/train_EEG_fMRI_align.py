# Training code for EEG-fMRI alignment
import torch
import torch.nn as nn

# Optimization
import torch.optim as optim
# cosine learning rate scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR

# import the model
from model.EEG_fMRI_align import EEG_fMRI_Align

# import the data loader
from data.EEG_fMRI_align_dataset import get_data_loader

# Tensor board for logging
from torch.utils.tensorboard import SummaryWriter

# progress bar
from tqdm import tqdm

# arg parser
import argparse

# import type hinting
import typing

# import utilities
from time import time
import os
import shutil

# get parsed arguments
def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-fMRI alignment model")
    
    # General training settings
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=1, help='cuda number (default: 1)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-3)')
    parser.add_argument('--multi_lr', type=lambda x: x.lower() == 'true', default=True,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--weight_decay', type=float, default=5e-2, help='weight decay (default: 1e-2)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='gradient clipping value (default: 1)')

    # Encoder backbone settings
    parser.add_argument('--backbone', type=str, default='CBraMod', choices=['CBraMod', 'ATMS'], help='EEG encoder backbone (default: CBraMod)')
    parser.add_argument('--frozen', type=lambda x: x.lower() == 'true',
                        default=False, help='frozen')  # whether to freeze the EEG encoder during training
    parser.add_argument('--use_pretrained_weights', type=lambda x: x.lower() == 'true',
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default=None,
                        help='CBraMod foundation model dir for loading pretrained weights (default: None)')
    
    # ATMS backbone settings
    parser.add_argument('--atms_emb_size', type=int, default=40, help='ATMS embedding size (default: 40)')
    parser.add_argument('--out_mlp_dim', type=int, default=3200, help='ATMS projection dimension (default: 3200)')
    parser.add_argument('--atms_drop_proj', type=float, default=0.5, help='ATMS projection head dropout (default: 0.5)')
    parser.add_argument('--atms_d_model', type=int, default=250, help='ATMS iTransformer d_model (default: 250)')
    parser.add_argument('--atms_n_heads', type=int, default=4, help='ATMS iTransformer number of heads (default: 4)')
    parser.add_argument('--atms_d_ff', type=int, default=256, help='ATMS iTransformer feedforward dimension (default: 256)')
    parser.add_argument('--atms_dropout', type=float, default=0.25, help='ATMS iTransformer dropout (default: 0.25)')
    parser.add_argument('--atms_factor', type=int, default=1, help='ATMS iTransformer factor (default: 1)')

    # Checkpoints (full checkpoint loading - please don't specify if you used foundation weights)
    parser.add_argument('--model_dir', type=str, default=None, help='full model dir for loading (default: None)')
    parser.add_argument('--script_path', type=str, default=None, help='path to the bash script used to launch training (saved for reproducibility)')

    # Dataset settings
    parser.add_argument('--datasets_dir', type=str,
                        default=None,
                        help='datasets_dir')
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    
    # EEG-fMRI alignment (model) specific parameters
    parser.add_argument("--mse_scale", type=float, default=1.0, help="MSE loss scale")
    parser.add_argument("--infonce_scale", type=float, default=1.0, help="InfoNCE loss scale")
    parser.add_argument("--proto_distill_scale", type=float, default=1.0, help="Prototypical distillation loss scale")
    parser.add_argument("--temperature", type=float, default=0.07, help="InfoNCE temperature")
    parser.add_argument("--normalize_fmri", type=lambda x: x.lower() == 'true', default=True, help="Normalize fMRI to unit norm")
    parser.add_argument("--alignment_attention_heads", type=int, default=4, help="Number of attention heads for alignment attention")
    parser.add_argument('--alignment_attention_dropout', type=float, default=0.25, help='Dropout for alignment attention')
    
    # WARNING: These parameters are baked in CBraMod module
    # TODO: refactor and pull these out of CBraMod backbone
    parser.add_argument("--pooling_type", type=str, default='flatten', choices=['flatten', 'attention', 'multitoken_vit'],
                        help="Pooling type: flatten, attention, or multitoken_vit")
    parser.add_argument("--embedding_dim", type=int, default=4096, help="Output embedding dimension")
    parser.add_argument("--mlp_layers", type=int, default=2, help="Number of MLP layers")
    parser.add_argument("--attention_heads", type=int, default=8, help="Number of attention heads for attention pooling")
    parser.add_argument("--num_tokens", type=int, default=4, help="Number of learnable tokens for multitoken_vit pooling")
    parser.add_argument("--num_transformer_layers", type=int, default=4, help="Number of transformer layers for multitoken_vit pooling")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads for multitoken_vit transformer")
   
    return parser.parse_args()

# training loop
def train(model: EEG_fMRI_Align,
            data_loader: typing.Dict[str, torch.utils.data.DataLoader],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler._LRScheduler,
            device: torch.device,
            num_epochs: int,
            ckpt_dir: str,
            clip_value: float = -1.0,
            logger: SummaryWriter = None):
        
        """
        Training loop for EEG-fMRI alignment model
        Args:
            model: EEG-fMRI alignment model
            data_loader: dictionary of data loaders for 'train', 'val', and 'test' sets
            optimizer: optimizer for training
            scheduler: learning rate scheduler (per step)
            device: device to train on
            num_epochs: number of epochs to train for
            ckpt_dir: directory to save checkpoints
            clip_value: value for gradient clipping (default: -1, no clipping)
            logger: summary writer for logging
        Returns:
            None
        """
    
        best_val_loss = float('inf')
    
        for epoch in range(num_epochs):

            # Set to training mode
            model.train()

            # accumulate losses for logging
            total_losses = []
            mse_losses = []
            infonce_losses = []
            proto_losses = []
    
            # Data loaded from dataset: 
            # 'eeg', 'fmri', 'label', 'things_img_idx', 'nsd_img_idx'
            for EEG, fMRI, label, _ , _ in tqdm(data_loader['train'], desc = f"Epoch {epoch+1}/{num_epochs}"):
                
                # Move data to device
                EEG, fMRI, label = EEG.to(device), fMRI.to(device), label.to(device)
    
                # Zero gradients
                optimizer.zero_grad()

                # Forward pass and compute loss
                out = model.forward(EEG, fMRI)
                loss, mse_loss, infonce_loss, proto_loss = model.calc_alignment_loss(out.squeeze(), fMRI.squeeze(), label)

                # Backward pass
                loss.backward()
    
                # Gradient clipping
                if clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
    
                # Update parameters
                optimizer.step()

                # Schedule step
                scheduler.step()

                # Accumulate losses for logging
                total_losses.append(loss.item())
                mse_losses.append(mse_loss.item())
                infonce_losses.append(infonce_loss.item())
                proto_losses.append(proto_loss.item())
            
            # end epoch training, compute average losses
            # verbose
            avg_train_loss = sum(total_losses) / len(total_losses)
            avg_mse_loss = sum(mse_losses) / len(mse_losses)
            avg_infonce_loss = sum(infonce_losses) / len(infonce_losses)
            avg_proto_loss = sum(proto_losses) / len(proto_losses)
            print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} (MSE: {avg_mse_loss:.4f}, InfoNCE: {avg_infonce_loss:.4f}, Proto: {avg_proto_loss:.4f})")
            
            # Validation - on training metrics
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                mse_val_loss = 0.0
                infonce_val_loss = 0.0
                proto_val_loss = 0.0

                for EEG, fMRI, label, _ , _ in data_loader['val']:
                    EEG, fMRI, label = EEG.to(device), fMRI.to(device), label.to(device)
                    out = model.forward(EEG, fMRI)
                    loss, mse_loss, infonce_loss, proto_loss = model.calc_alignment_loss(out.squeeze(), fMRI.squeeze(), label)
                    val_loss += loss.item()
                    mse_val_loss += mse_loss.item()
                    infonce_val_loss += infonce_loss.item()
                    proto_val_loss += proto_loss.item()

                avg_val_loss = val_loss / len(data_loader['val'])
                avg_proto_val_loss = proto_val_loss / len(data_loader['val'])
                avg_mse_val_loss = mse_val_loss / len(data_loader['val'])
                avg_infonce_val_loss = infonce_val_loss / len(data_loader['val'])

            # Validation - on val metrics
            outputs = []
            fMRI_targets = []
            val_loss = 0.0
            with torch.no_grad():
                model.eval()
                for EEG, fMRI, label, _ , _ in tqdm(data_loader['val'], desc=f"Validation Epoch {epoch+1}"):

                    # Move data to device
                    EEG, fMRI = EEG.to(device), fMRI.to(device)

                    # Forward pass and compute loss
                    out = model.forward(EEG, fMRI)
                    outputs.append(out)
                    fMRI_targets.append(fMRI)
                    loss, _, _, _ = model.calc_alignment_loss(out.squeeze(), fMRI.squeeze(), label)
                    val_loss += loss.item()

                # concatenate outputs and compute metrics
                outputs = torch.cat(outputs, dim=0)
                fMRI_targets = torch.cat(fMRI_targets, dim=0)
                mse, cos_sim, retrieval_acc, retrieval_acc_top10 = model.get_metrics_for_alignment(outputs.squeeze(), fMRI_targets.squeeze())
            # verbose validation metrics
            avg_val_loss = val_loss / len(data_loader['val'])
            print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f} (MSE: {mse:.4f}, CosSim: {cos_sim:.4f}, Retrieval Acc (Top1): {retrieval_acc:.4f}, Top10: {retrieval_acc_top10:.4f})")
    
            # Checkpointing
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # delete every old checkpoint in the ckpt_dir
                for filename in os.listdir(ckpt_dir):
                    file_path = os.path.join(ckpt_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                # save new ckpt
                checkpoint_path = f"{ckpt_dir}/best_model_epoch_{epoch+1}.pth"
                model.save_model(checkpoint_path)
                print(f"Saved best model to {checkpoint_path}")

            # get optim state
            optim_state = optimizer.state_dict()

            # Tensor board logging
            if logger is not None:
                logger.add_scalar('Loss/Train', avg_train_loss, epoch)
                logger.add_scalar('Loss/Val', avg_val_loss, epoch)
                logger.add_scalar('MSE_Loss/Train', avg_mse_loss, epoch)
                logger.add_scalar('MSE_Loss/Val', avg_mse_val_loss, epoch)
                logger.add_scalar('InfoNCE_Loss/Train', avg_infonce_loss, epoch)
                logger.add_scalar('InfoNCE_Loss/Val', avg_infonce_val_loss, epoch)
                logger.add_scalar('Proto_Distill_Loss/Train', avg_proto_loss, epoch)
                logger.add_scalar('Proto_Distill_Loss/Val', avg_proto_val_loss, epoch)
                logger.add_scalar('Metrics/MSE', mse, epoch)
                logger.add_scalar('Metrics/Cosine_Similarity', cos_sim, epoch)
                logger.add_scalar('Metrics/Retrieval_Accuracy_Top1', retrieval_acc, epoch)
                logger.add_scalar('Metrics/Retrieval_Accuracy_Top10', retrieval_acc_top10, epoch)
                # Log optimizer learning rate
                param_groups = optim_state['param_groups']
                for i, group in enumerate(param_groups):
                    logger.add_scalar(f'Learning_Rate/Group_{i}', group['lr'], epoch)

# testing loop
def test(model: EEG_fMRI_Align,
         data_loader: torch.utils.data.DataLoader,
         device: torch.device,
         logger: SummaryWriter = None):
    """
    Test loop for EEG-fMRI alignment model
    Args:
        model: EEG-fMRI alignment model
        data_loader: data loader for test set
        device: device to test on
        logger: summary writer for logging
    Returns:
        None
    """
    model.eval()
    outputs = []
    fMRI_targets = []
    with torch.no_grad():

        for EEG, fMRI, _ , _ , _ in tqdm(data_loader, desc = "Testing"):

            # Move data to device
            EEG, fMRI = EEG.to(device), fMRI.to(device)

            # Forward pass
            out = model.forward(EEG, fMRI)
            outputs.append(out)
            fMRI_targets.append(fMRI)

        # concatenate outputs and compute metrics
        outputs = torch.cat(outputs, dim=0)
        fMRI_targets = torch.cat(fMRI_targets, dim=0)
        mse, cos_sim, retrieval_acc, retrieval_acc_top10 = model.get_metrics_for_alignment(outputs.squeeze(), fMRI_targets.squeeze())
        print(f"Test Metrics - MSE: {mse:.4f}, CosSim: {cos_sim:.4f}, Retrieval Acc (Top1): {retrieval_acc:.4f}, Top10: {retrieval_acc_top10:.4f}")

        # Tensor board logging
        if logger is not None:
            logger.add_scalar('Test/MSE', mse)
            logger.add_scalar('Test/Cosine_Similarity', cos_sim)
            logger.add_scalar('Test/Retrieval_Accuracy_Top1', retrieval_acc)
            logger.add_scalar('Test/Retrieval_Accuracy_Top10', retrieval_acc_top10)

# Main function
if __name__ == "__main__":

    # Get arguments
    args = get_args()
    # verbose arguments
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Get device
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment folder
    log_dir = f"runs/EEG_fMRI_align_{int(time())}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    # Create logging and ckpt directories
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir = f"{log_dir}/checkpoints"
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Save the launch script for reproducibility
    if args.script_path and os.path.isfile(args.script_path):
        shutil.copy(args.script_path, f"{log_dir}/run_script.sh")
        print(f"Saved launch script to {log_dir}/run_script.sh")
    
    # Create tensorboard logger
    logger = SummaryWriter(log_dir=tensorboard_dir)

    # Get data loaders
    data_loader = get_data_loader(args.datasets_dir, batch_size = args.batch_size, normalize_fmri = args.normalize_fmri)

    # Create model config dictionary
    # depends on the backbone
    if args.backbone == 'CBraMod':
        encoder_config = {
            'encoder_type': 'CBraMod',
            'cuda' : args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
            'pooling_type': args.pooling_type,
            'attention_heads': args.attention_heads,
            'num_tokens': args.num_tokens,
            'num_transformer_layers': args.num_transformer_layers,
            'num_attention_heads': args.num_attention_heads,
            'mlp_layers': args.mlp_layers,
            'embedding_dim': args.embedding_dim
        }
    elif args.backbone == 'ATMS':
        encoder_config = {
            'encoder_type': 'ATMS',
            'cuda' : args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
            "num_channels": 63,
            "seq_len":      250,
            "emb_size":     args.atms_emb_size,
            "proj_dim":     1024,
            "drop_proj":    args.atms_drop_proj,
            "d_model":      args.atms_d_model,
            "n_heads":      args.atms_n_heads,
            "e_layers":     1,
            "d_ff":         args.atms_d_ff,
            "dropout":      args.atms_dropout,
            "factor":       args.atms_factor,
            "out_mlp_dim":  args.out_mlp_dim
        }
    else:
        raise ValueError(f"Unsupported backbone type: {args.backbone}")
    # Build model config
    model_config = {
        'EEG_Encoder': encoder_config,
        'Loss': {
            'mse_scale': args.mse_scale,
            'infonce_scale': args.infonce_scale,
            'proto_distill_scale': args.proto_distill_scale,
            'temperature': args.temperature,
            'normalize_fmri': args.normalize_fmri
        },
        'Attention_Merge': {
            'alignment_attention_heads': args.alignment_attention_heads,
            'alignment_attention_dropout': args.alignment_attention_dropout
        }
    }

    # Create model instance
    model = EEG_fMRI_Align(model_config).to(device)

    # Load full model checkpoint if specified (overrides foundation weights)
    if args.model_dir is not None:
        model.load_model(args.model_dir, device)
        print(f"Loaded model from {args.model_dir}")
    # Freeze EEG encoder if specified
    if args.frozen:
        for param in model.eeg_encoder.parameters():
            param.requires_grad = False
        print("Frozen EEG encoder parameters")

    # Create optimizer
    # model.eeg_encoder.backbone.parameters() 
    # other than model.eeg_encoder.backbone.parameters() 
    # model.transformer.parameters()
    # choose a optimizer get attributes by name from torch.optim
    optimizer_class: torch.optim.Optimizer = getattr(optim, args.optimizer)
    print(f"Using optimizer: {optimizer_class.__name__}")
    # Get parameters
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'eeg_encoder.backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    print(f"Backbone parameters: {len(backbone_params)}, Other parameters: {len(other_params)}")
    # Set learning rates
    if args.multi_lr:
        optimizer = optimizer_class([
            {'params': backbone_params, 'lr': args.lr},
            {'params': other_params, 'lr': args.lr * 5}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Create learning rate scheduler (per step)
    total_steps = args.epochs * len(data_loader['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Train the model
    train(model, data_loader, optimizer, scheduler, device, args.epochs, ckpt_dir, args.clip_value, logger)

    # Test the model on the test set
    test(model, data_loader['test'], device, logger)

    # close the tensorboard logger
    logger.close()
    print("Training complete. Tensorboard logs saved to:", tensorboard_dir)
