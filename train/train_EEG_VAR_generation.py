"""
EEG-VAR Generation Training Script

Stage 2 VAR-based image generation training.
Loads frozen stage 1 EEG-CLIP alignment model and trains VAR transformer.

Key features:
- Per-scale loss weighting to prevent high-resolution scale dominance
- Online VAE tokenization during training
- Classifier-free guidance training with 10% dropout
- Cosine annealing learning rate schedule with warmup
"""

import os
import argparse
import time
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model.EEG_VAR_generation import EEG_VAR_Generation
from data.EEG_VAR_generation_dataset import EEG_VAR_Generation_Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Train EEG-VAR Generation Model')

    # Paths
    parser.add_argument('--stage1_ckpt', type=str, required=True,
                        help='Path to stage 1 checkpoint')
    parser.add_argument('--lmdb_dir', type=str, required=True,
                        help='Path to LMDB directory')
    parser.add_argument('--h5_path', type=str, required=True,
                        help='Path to HDF5 image file')
    parser.add_argument('--vae_ckpt', type=str, default='pretrained/vae_ch160v4096z32.pth',
                        help='Path to VQVAE checkpoint')
    parser.add_argument('--var_ckpt', type=str, default=None,
                        help='Path to VAR checkpoint (default: pretrained/var_d{depth}.pth)')

    # Model config
    parser.add_argument('--eeg_encoder_type', type=str, default='CBraMod',
                        choices=['CBraMod', 'ATMS'],
                        help='EEG encoder type')
    parser.add_argument('--var_depth', type=int, default=16,
                        help='VAR transformer depth')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--freeze_eeg_encoder',
                        type=lambda x: x.lower() == 'true',
                        default=True,
                        help='Freeze EEG encoder (stage 1) during training (default: True)')
    parser.add_argument('--eeg_encoder_lr_scale',
                        type=float,
                        default=0.1,
                        help='LR multiplier for EEG encoder when unfrozen (default: 0.1)')
    parser.add_argument('--unfrozen_eeg_mode',
                        type=str,
                        default='eval',
                        choices=['eval', 'train'],
                        help='Mode for unfrozen EEG encoder: "eval" (partial fine-tuning, BN/Dropout frozen) or "train" (full fine-tuning, BN/Dropout active). Only applies when freeze_eeg_encoder=False. Default: eval (recommended)')

    # Data loading
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')

    # Logging and checkpointing
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--output_dir', type=str, default='runs',
                        help='Output directory for checkpoints and logs')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def get_stage1_config(eeg_encoder_type: str) -> dict:
    """Get stage 1 model configuration."""
    if eeg_encoder_type == 'CBraMod':
        return {
            'EEG_Encoder': {
                'type': 'CBraMod_EEG_Encoder',
                'pooling_type': 'attention',
                'output_dim': 1024
            }
        }
    elif eeg_encoder_type == 'ATMS':
        return {
            'EEG_Encoder': {
                'type': 'ATMS_EEG_Encoder',
                'output_dim': 1024
            }
        }
    else:
        raise ValueError(f"Unknown EEG encoder type: {eeg_encoder_type}")


def get_var_config(depth: int) -> dict:
    """Get VAR model configuration."""
    # Follow AVDE's drop_path_rate calculation: dpr = 0.1 * depth / 24
    # This provides depth-adaptive stochastic depth regularization
    # Examples: depth=16 -> 0.0667, depth=24 -> 0.1, depth=30 -> 0.125
    drop_path_rate = 0.1 * depth / 24

    return {
        'depth': depth,
        'embed_dim': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'drop_rate': 0.0,
        'attn_drop_rate': 0.0,
        'drop_path_rate': drop_path_rate,  # Dynamic calculation based on depth
        'norm_eps': 1e-6,
        'shared_aln': False,
        'cond_drop_rate': 0.1,  # CFG dropout
        'attn_l2_norm': False,
        'patch_nums': (1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 scales
        'flash_if_available': True,
        'fused_if_available': True,
        'vocab_size': 4096,
        'z_channels': 32,
        'ch': 160,
    }


def create_optimizer(model: nn.Module, args):
    """Create optimizer with weight decay exclusions and differential LR."""
    # Exclude bias and LayerNorm parameters from weight decay
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'norm.weight', 'norm.bias']

    if args.freeze_eeg_encoder:
        # Only optimize VAR parameters
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.var.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.var.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
    else:
        # Optimize both VAR and EEG encoder with differential LR
        eeg_lr = args.lr * args.eeg_encoder_lr_scale
        print(f"VAR LR: {args.lr:.2e}, EEG encoder LR: {eeg_lr:.2e} (scale: {args.eeg_encoder_lr_scale})")
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.var.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.var.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': args.lr,
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in model.eeg_clip_model.named_parameters() if not any(nd in n for nd in no_decay)],
                'lr': eeg_lr,
                'weight_decay': args.weight_decay
            },
            {
                'params': [p for n, p in model.eeg_clip_model.named_parameters() if any(nd in n for nd in no_decay)],
                'lr': eeg_lr,
                'weight_decay': 0.0
            }
        ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95))
    return optimizer


def create_scheduler(optimizer, num_training_steps: int, warmup_steps: int):
    """Create cosine annealing scheduler with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_epoch(model, train_loader, optimizer, scheduler, epoch, args, writer, global_step):
    """Train for one epoch."""
    model.train()
    model.var.train()  # Ensure VAR is in training mode

    # Control EEG encoder state based on freeze_eeg_encoder and unfrozen_eeg_mode
    if args.freeze_eeg_encoder:
        # Fully frozen: no gradients, eval mode
        model.eeg_clip_model.eval()
    else:
        # Unfrozen: gradients enabled, but mode depends on unfrozen_eeg_mode
        if args.unfrozen_eeg_mode == 'eval':
            # Partial fine-tuning: update weights but keep BN/Dropout frozen
            model.eeg_clip_model.eval()
            if epoch == 1:  # Only print once
                print(f"EEG encoder in EVAL mode (partial fine-tuning: weights updated, BN/Dropout frozen)")
        else:  # 'train'
            # Full fine-tuning: update weights AND BN/Dropout statistics
            model.eeg_clip_model.train()
            if epoch == 1:  # Only print once
                print(f"EEG encoder in TRAIN mode (full fine-tuning: weights + BN/Dropout updated)")

    model.vae.eval()  # Keep VQVAE frozen

    total_loss = 0.0
    start_time = time.time()

    for batch_idx, (eeg, images, _) in enumerate(train_loader):
        eeg = eeg.to(args.device)
        images = images.to(args.device)

        # Forward pass
        logits_BLV, gt_idx_Bl = model(eeg, images)

        # Compute loss with per-scale weighting
        loss = model.compute_loss(logits_BLV, gt_idx_Bl, label_smoothing=args.label_smoothing)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.var.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        global_step += 1

        # Logging
        if batch_idx % args.log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time

            # Compute per-scale metrics
            per_scale_losses, per_scale_accs = model.compute_per_scale_metrics(
                logits_BLV, gt_idx_Bl, label_smoothing=args.label_smoothing
            )

            # Compute top-k accuracies
            top5_acc = model.compute_topk_accuracy(logits_BLV, gt_idx_Bl, k=5)

            # Compute global token-level accuracy
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            pred_BL = torch.argmax(logits_BLV, dim=-1)
            # Slice pred_BL to only keep the generated image tokens (ignore prefix)
            pred_BL_tokens = pred_BL[:, -gt_BL.shape[1]:]
            global_acc = (pred_BL_tokens == gt_BL).float().mean().item()

            # Compute scale-mean accuracy
            scale_mean_acc = sum(per_scale_accs) / len(per_scale_accs)

            # Compute gradient norm (using PyTorch built-in)
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float('inf')
            ).item()

            print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) '
                  f'Acc: {global_acc:.3f} '
                  f'LR: {lr:.6f} '
                  f'Time: {elapsed:.2f}s')

            # TensorBoard logging
            writer.add_scalar('Train/Loss_Total', loss.item(), global_step)
            writer.add_scalar('Train/LR', lr, global_step)
            writer.add_scalar('Train/Grad_Norm', total_norm, global_step)
            writer.add_scalar('Train/Accuracy_Global_Token', global_acc, global_step)
            writer.add_scalar('Train/Accuracy_Scale_Mean', scale_mean_acc, global_step)
            writer.add_scalar('Train/Accuracy_Top5', top5_acc, global_step)

            # Log per-scale metrics
            patch_nums = model.var.patch_nums
            for scale_idx, (pn, scale_loss, scale_acc) in enumerate(
                zip(patch_nums, per_scale_losses, per_scale_accs)
            ):
                writer.add_scalar(
                    f'Train/Loss_Scale{scale_idx}_{pn}x{pn}',
                    scale_loss,
                    global_step
                )
                writer.add_scalar(
                    f'Train/Acc_Scale{scale_idx}_{pn}x{pn}',
                    scale_acc,
                    global_step
                )

    avg_loss = total_loss / len(train_loader)
    return avg_loss, global_step


def validate(model, val_loader, args, writer=None, epoch=None):
    """Validate the model with comprehensive metrics."""
    model.eval()
    total_loss = 0.0

    # Accumulate per-scale metrics
    patch_nums = model.var.patch_nums
    num_scales = len(patch_nums)
    accumulated_scale_losses = [0.0] * num_scales
    accumulated_scale_accs = [0.0] * num_scales

    # Accumulate global metrics
    total_correct_tokens = 0
    total_tokens = 0
    total_top5_acc = 0.0
    total_top10_acc = 0.0
    total_perplexity = 0.0
    total_entropy = 0.0

    num_batches = 0

    with torch.no_grad():
        for eeg, images, _ in val_loader:
            eeg = eeg.to(args.device)
            images = images.to(args.device)

            # Forward pass
            logits_BLV, gt_idx_Bl = model(eeg, images)

            # Compute total loss
            loss = model.compute_loss(
                logits_BLV, gt_idx_Bl, label_smoothing=0.0
            )
            total_loss += loss.item()

            # Compute per-scale metrics
            per_scale_losses, per_scale_accs = model.compute_per_scale_metrics(
                logits_BLV, gt_idx_Bl, label_smoothing=0.0
            )
            for i in range(num_scales):
                accumulated_scale_losses[i] += per_scale_losses[i]
                accumulated_scale_accs[i] += per_scale_accs[i]

            # Compute global token-level accuracy
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            pred_BL = torch.argmax(logits_BLV, dim=-1)
            # Slice pred_BL to only keep the generated image tokens (ignore prefix)
            pred_BL_tokens = pred_BL[:, -gt_BL.shape[1]:]
            total_correct_tokens += (pred_BL_tokens == gt_BL).sum().item()
            total_tokens += gt_BL.shape[0] * gt_BL.shape[1]

            # Compute top-k accuracies
            total_top5_acc += model.compute_topk_accuracy(
                logits_BLV, gt_idx_Bl, k=5
            )
            total_top10_acc += model.compute_topk_accuracy(
                logits_BLV, gt_idx_Bl, k=10
            )

            # Compute perplexity and entropy
            total_perplexity += model.compute_perplexity(logits_BLV, gt_idx_Bl)
            total_entropy += model.compute_prediction_entropy(logits_BLV)

            num_batches += 1

    # Compute averages
    avg_loss = total_loss / num_batches
    avg_scale_losses = [sl / num_batches for sl in accumulated_scale_losses]
    avg_scale_accs = [sa / num_batches for sa in accumulated_scale_accs]

    # Global token-level accuracy
    global_token_acc = total_correct_tokens / total_tokens

    # Scale-mean accuracy (unweighted average)
    scale_mean_acc = sum(avg_scale_accs) / len(avg_scale_accs)

    avg_top5_acc = total_top5_acc / num_batches
    avg_top10_acc = total_top10_acc / num_batches
    avg_perplexity = total_perplexity / num_batches
    avg_entropy = total_entropy / num_batches

    # Compute scale loss ratio (high-res vs low-res)
    loss_ratio = avg_scale_losses[-1] / (avg_scale_losses[0] + 1e-10)

    # TensorBoard logging
    if writer is not None and epoch is not None:
        writer.add_scalar('Val/Loss_Total', avg_loss, epoch)
        writer.add_scalar('Val/Accuracy_Global_Token', global_token_acc, epoch)
        writer.add_scalar('Val/Accuracy_Scale_Mean', scale_mean_acc, epoch)
        writer.add_scalar('Val/Accuracy_Top5', avg_top5_acc, epoch)
        writer.add_scalar('Val/Accuracy_Top10', avg_top10_acc, epoch)
        writer.add_scalar('Val/Perplexity', avg_perplexity, epoch)
        writer.add_scalar('Val/Prediction_Entropy', avg_entropy, epoch)
        writer.add_scalar('Val/Loss_Ratio_High_vs_Low', loss_ratio, epoch)

        # Log per-scale metrics
        for scale_idx, (pn, scale_loss, scale_acc) in enumerate(
            zip(patch_nums, avg_scale_losses, avg_scale_accs)
        ):
            writer.add_scalar(
                f'Val/Loss_Scale{scale_idx}_{pn}x{pn}',
                scale_loss,
                epoch
            )
            writer.add_scalar(
                f'Val/Acc_Scale{scale_idx}_{pn}x{pn}',
                scale_acc,
                epoch
            )

    return avg_loss, global_token_acc


def main():
    args = parse_args()

    # Set default VAR checkpoint path if not provided
    if args.var_ckpt is None:
        args.var_ckpt = f'pretrained/var_d{args.var_depth}.pth'

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f'EEG_VAR_generation_{args.eeg_encoder_type}_d{args.var_depth}_{timestamp}'
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    print(f"Output directory: {output_dir}")
    print(f"Arguments: {json.dumps(vars(args), indent=2)}")

    # Create datasets
    print("Creating datasets...")
    train_dataset = EEG_VAR_Generation_Dataset(
        lmdb_dir=args.lmdb_dir,
        h5_path=args.h5_path,
        mode='train',
        image_size=256
    )
    val_dataset = EEG_VAR_Generation_Dataset(
        lmdb_dir=args.lmdb_dir,
        h5_path=args.h5_path,
        mode='val',
        image_size=256
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    print("Creating model...")
    stage1_config = get_stage1_config(args.eeg_encoder_type)
    var_config = get_var_config(args.var_depth)

    model = EEG_VAR_Generation(
        stage1_config=stage1_config,
        stage1_ckpt_path=args.stage1_ckpt,
        vae_ckpt_path=args.vae_ckpt,
        var_ckpt_path=args.var_ckpt,
        var_config=var_config,
        freeze_eeg_encoder=args.freeze_eeg_encoder,
        device=args.device
    )
    model = model.to(args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.var.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (VAR only): {trainable_params:,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args)
    num_training_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = create_scheduler(optimizer, num_training_steps, warmup_steps)

    print(f"Training steps: {num_training_steps}, Warmup steps: {warmup_steps}")

    # Training loop
    best_val_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*80}")

        # Train
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, epoch, args, writer, global_step
        )
        print(f"Train loss: {train_loss:.4f}")
        writer.add_scalar('epoch/train_loss', train_loss, epoch)

        # Validate
        if epoch % args.val_interval == 0:
            val_loss, val_acc = validate(
                model, val_loader, args, writer=writer, epoch=epoch
            )
            print(f"Val loss: {val_loss:.4f}, Val acc (global): {val_acc:.3f}")

            # Save best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint = {
                    'epoch': epoch,
                    'var_state_dict': model.var.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'args': vars(args)
                }

                # If EEG encoder was unfrozen, save its weights too
                if not args.freeze_eeg_encoder:
                    checkpoint['eeg_clip_state_dict'] = model.eeg_clip_model.state_dict()
                    print("Saved fine-tuned EEG encoder weights to checkpoint")

                torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', f'best_model_epoch_{epoch}.pth'))
                print(f"Saved best model at epoch {epoch} (val_loss: {val_loss:.4f})")

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'var_state_dict': model.var.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss if epoch % args.val_interval == 0 else None,
                'args': vars(args)
            }

            if not args.freeze_eeg_encoder:
                checkpoint['eeg_clip_state_dict'] = model.eeg_clip_model.state_dict()

            torch.save(checkpoint, os.path.join(output_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth'))
            print(f"Saved checkpoint at epoch {epoch}")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Output directory: {output_dir}")

    writer.close()


if __name__ == '__main__':
    main()

