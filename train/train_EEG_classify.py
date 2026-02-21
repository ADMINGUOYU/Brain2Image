# Training code for EEG classification (K-means cluster labels)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from model.EEG_classify import EEG_Classify
from data.EEG_fMRI_align_dataset import get_data_loader

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import typing
from time import time
import os
import shutil


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG classification model")

    # General training settings
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--multi_lr', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--clip_value', type=float, default=1.0)

    # Encoder backbone settings
    parser.add_argument('--backbone', type=str, default='CBraMod', choices=['CBraMod', 'ATMS'])
    parser.add_argument('--frozen', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_pretrained_weights', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--foundation_dir', type=str, default=None)
    parser.add_argument('--model_dir', type=str, default=None)
    parser.add_argument('--script_path', type=str, default=None)

    # Dataset settings
    parser.add_argument('--datasets_dir', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--normalize_fmri', type=lambda x: x.lower() == 'true', default=True)

    # ATMS backbone settings
    parser.add_argument('--atms_emb_size', type=int, default=40)
    parser.add_argument('--atms_drop_proj', type=float, default=0.5)
    parser.add_argument('--atms_d_model', type=int, default=250)
    parser.add_argument('--atms_n_heads', type=int, default=4)
    parser.add_argument('--atms_d_ff', type=int, default=256)
    parser.add_argument('--atms_dropout', type=float, default=0.25)
    parser.add_argument('--atms_factor', type=int, default=1)

    # Classification head settings
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--mlp_layers', type=int, default=2)

    return parser.parse_args()


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """Returns overall accuracy and per-class accuracy tensor."""
    preds = logits.argmax(dim=1)
    correct = (preds == labels).float()
    overall_acc = correct.mean().item()

    per_class_acc = torch.zeros(num_classes)
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() > 0:
            per_class_acc[c] = correct[mask].mean().item()
        else:
            per_class_acc[c] = float('nan')
    return overall_acc, per_class_acc


def train(model: EEG_Classify,
          data_loader: typing.Dict[str, torch.utils.data.DataLoader],
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          device: torch.device,
          num_epochs: int,
          ckpt_dir: str,
          clip_value: float = -1.0,
          logger: SummaryWriter = None):

    best_val_loss = float('inf')
    num_classes = model.num_classes

    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        all_logits = []
        all_labels = []

        for EEG, fMRI, label, _, _ in tqdm(data_loader['train'], desc=f"Epoch {epoch+1}/{num_epochs}"):
            EEG, label = EEG.to(device), label.to(device)

            optimizer.zero_grad()
            logits = model.forward(EEG)
            loss = model.calc_loss(logits, label)
            loss.backward()

            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            scheduler.step()

            train_losses.append(loss.item())
            all_logits.append(logits.detach())
            all_labels.append(label.detach())

        # Train metrics
        avg_train_loss = sum(train_losses) / len(train_losses)
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        train_acc, train_per_class = compute_accuracy(all_logits, all_labels, num_classes)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f}")
        print(f"  Per-class Acc: {[f'{v:.4f}' for v in train_per_class.tolist()]}")

        # Validation
        model.eval()
        val_losses = []
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for EEG, fMRI, label, _, _ in tqdm(data_loader['val'], desc=f"Val Epoch {epoch+1}"):
                EEG, label = EEG.to(device), label.to(device)
                logits = model.forward(EEG)
                loss = model.calc_loss(logits, label)
                val_losses.append(loss.item())
                val_logits.append(logits)
                val_labels.append(label)

        avg_val_loss = sum(val_losses) / len(val_losses)
        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_acc, val_per_class = compute_accuracy(val_logits, val_labels, num_classes)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
        print(f"  Per-class Acc: {[f'{v:.4f}' for v in val_per_class.tolist()]}")

        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            for filename in os.listdir(ckpt_dir):
                file_path = os.path.join(ckpt_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            checkpoint_path = f"{ckpt_dir}/best_model_epoch_{epoch+1}.pth"
            model.save_model(checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")

        # TensorBoard logging
        if logger is not None:
            logger.add_scalar('Loss/Train', avg_train_loss, epoch)
            logger.add_scalar('Loss/Val', avg_val_loss, epoch)
            logger.add_scalar('Accuracy/Train', train_acc, epoch)
            logger.add_scalar('Accuracy/Val', val_acc, epoch)
            for c in range(num_classes):
                logger.add_scalar(f'PerClassAcc_Train/Class_{c}', train_per_class[c].item(), epoch)
                logger.add_scalar(f'PerClassAcc_Val/Class_{c}', val_per_class[c].item(), epoch)
            param_groups = optimizer.state_dict()['param_groups']
            for i, group in enumerate(param_groups):
                logger.add_scalar(f'Learning_Rate/Group_{i}', group['lr'], epoch)


def test(model: EEG_Classify,
         data_loader: torch.utils.data.DataLoader,
         device: torch.device,
         logger: SummaryWriter = None):

    model.eval()
    num_classes = model.num_classes
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for EEG, fMRI, label, _, _ in tqdm(data_loader, desc="Testing"):
            EEG, label = EEG.to(device), label.to(device)
            logits = model.forward(EEG)
            all_logits.append(logits)
            all_labels.append(label)

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        test_acc, per_class_acc = compute_accuracy(all_logits, all_labels, num_classes)
        test_loss = model.calc_loss(all_logits, all_labels).item()

    print(f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    print(f"  Per-class Acc: {[f'{v:.4f}' for v in per_class_acc.tolist()]}")

    if logger is not None:
        logger.add_scalar('Test/Loss', test_loss)
        logger.add_scalar('Test/Accuracy', test_acc)
        for c in range(num_classes):
            logger.add_scalar(f'Test/PerClassAcc/Class_{c}', per_class_acc[c].item())


if __name__ == "__main__":

    args = get_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create experiment folder
    log_dir = f"runs/EEG_classify_{int(time())}"
    os.makedirs(log_dir, exist_ok=True)
    print(f"Logging to {log_dir}")
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir = f"{log_dir}/checkpoints"
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    if args.script_path and os.path.isfile(args.script_path):
        shutil.copy(args.script_path, f"{log_dir}/run_script.sh")
        print(f"Saved launch script to {log_dir}/run_script.sh")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    # Get data loaders
    data_loader = get_data_loader(args.datasets_dir, batch_size=args.batch_size, normalize_fmri=args.normalize_fmri)

    # Build model config
    if args.backbone == 'CBraMod':
        encoder_config = {
            'encoder_type': 'CBraMod',
            'cuda': args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
        }
    elif args.backbone == 'ATMS':
        encoder_config = {
            'encoder_type': 'ATMS',
            'cuda': args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
            "num_channels": 63,
            "seq_len": 250,
            "emb_size": args.atms_emb_size,
            "proj_dim": 1024,
            "drop_proj": args.atms_drop_proj,
            "d_model": args.atms_d_model,
            "n_heads": args.atms_n_heads,
            "e_layers": 1,
            "d_ff": args.atms_d_ff,
            "dropout": args.atms_dropout,
            "factor": args.atms_factor,
        }
    else:
        raise ValueError(f"Unsupported backbone type: {args.backbone}")

    model_config = {
        'EEG_Encoder': encoder_config,
        'Classifier': {
            'num_classes': args.num_classes,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
            'mlp_layers': args.mlp_layers,
        }
    }

    model = EEG_Classify(model_config).to(device)

    # Load full model checkpoint if specified
    if args.model_dir is not None:
        model.load_model(args.model_dir, device)
        print(f"Loaded model from {args.model_dir}")

    # Freeze EEG encoder if specified
    if args.frozen:
        for param in model.eeg_encoder.parameters():
            param.requires_grad = False
        print("Frozen EEG encoder parameters")

    # Create optimizer with multi-LR
    optimizer_class: torch.optim.Optimizer = getattr(optim, args.optimizer)
    print(f"Using optimizer: {optimizer_class.__name__}")

    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'eeg_encoder.backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    print(f"Backbone parameters: {len(backbone_params)}, Other parameters: {len(other_params)}")

    if args.multi_lr:
        optimizer = optimizer_class([
            {'params': backbone_params, 'lr': args.lr},
            {'params': other_params, 'lr': args.lr * 5}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = args.epochs * len(data_loader['train'])
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # Train
    train(model, data_loader, optimizer, scheduler, device, args.epochs, ckpt_dir, args.clip_value, logger)

    # Test
    test(model, data_loader['test'], device, logger)

    logger.close()
    print("Training complete. Tensorboard logs saved to:", tensorboard_dir)
