# Training code for EEG-fMRI End-to-End generation pipeline
# MixCo is applied to 4096-dim EMBEDDINGS only, NEVER to raw EEG signals.
# All generation targets (CLIP bigG, VAE latents, ConvNeXt) are pre-computed.
# No frozen models (ViT-bigG, SD VAE, ConvNeXt) are loaded at training time.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from accelerate import Accelerator

from model.EEG_fMRI_e2e import EEG_fMRI_E2E
from data.EEG_fMRI_generation_e2e_dataset import get_generation_data_loader
from model.MindEYE2 import mixco, mixco_clip_target

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse
import typing
from time import time
import os
import shutil


def parse_and_validate_subject_weights(weight_str):
    """
    Parse subject weight string and validate that weights sum to 1.0.

    Args:
        weight_str: String in format "sub-01:0.4,sub-02:0.2,..." or None

    Returns:
        Dict mapping subject IDs to weights, or None if weight_str is None

    Raises:
        ValueError: If weights don't sum to 1.0
    """
    if weight_str is None:
        return None

    weights = {}
    for pair in weight_str.split(','):
        subject, weight = pair.strip().split(':')
        weights[subject.strip()] = float(weight.strip())

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Subject weights must sum to 1.0, got {total}")

    return weights


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-fMRI E2E generation model")

    # General training settings
    parser.add_argument('--seed', type=int, default=648)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-2)
    parser.add_argument('--clip_value', type=float, default=1.0)
    parser.add_argument('--use_amp', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--experiment_folder', type=str, default=None, help='experiment_folder: the main folder will under ./runs/experiment_folder/experiment_name_{timestamp}')
    parser.add_argument('--experiment_name', type=str, default='EEG_fMRI_e2e', help='experiment_name: the subfolder for this specific run, will be under the main experiment folder with timestamp, e.g. ./runs/experiment_folder/experiment_name_{timestamp}')
    parser.add_argument('--ckpt_interval', type=int, default=None, help='Number of epochs between saving checkpoints (default: None - save the best)')

    # Encoder backbone settings
    parser.add_argument('--backbone', type=str, default='CBraMod', choices=['CBraMod', 'ATMS'])
    parser.add_argument('--freeze_encoder', type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--freeze_mindeye2', type=lambda x: x.lower() == 'true', default=False) 
    parser.add_argument('--use_pretrained_weights', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--foundation_dir', type=str, default=None, help='Directory containing pretrained weights for the encoder backbone (CBraMod or ATMS)')

    # ATMS backbone settings
    parser.add_argument('--atms_emb_size', type=int, default=40)
    parser.add_argument('--out_mlp_dim', type=int, default=4096)
    parser.add_argument('--atms_drop_proj', type=float, default=0.5)
    parser.add_argument('--atms_d_model', type=int, default=250)
    parser.add_argument('--atms_n_heads', type=int, default=4)
    parser.add_argument('--atms_d_ff', type=int, default=256)
    parser.add_argument('--atms_dropout', type=float, default=0.25)
    parser.add_argument('--atms_factor', type=int, default=1)

    # CBraMod-specific parameters
    parser.add_argument("--pooling_type", type=str, default='flatten',
                        choices=['flatten', 'attention', 'multitoken_vit'])
    parser.add_argument("--embedding_dim", type=int, default=4096)
    parser.add_argument("--mlp_layers", type=int, default=2)
    parser.add_argument("--attention_heads", type=int, default=8)
    parser.add_argument("--num_tokens", type=int, default=4)
    parser.add_argument("--num_transformer_layers", type=int, default=4)
    parser.add_argument("--num_attention_heads", type=int, default=4)

    # Alignment loss parameters
    parser.add_argument("--mse_scale", type=float, default=1.0)
    parser.add_argument("--infonce_scale", type=float, default=0.2)
    parser.add_argument("--proto_distill_scale", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--normalize_fmri", type=lambda x: x.lower() == 'true', default=True)

    # E2E generation loss parameters
    parser.add_argument("--align_scale", type=float, default=1.0)
    parser.add_argument("--prior_scale", type=float, default=30.0)
    parser.add_argument("--clip_loss_scale", type=float, default=1.0)
    parser.add_argument("--blur_scale", type=float, default=0.5)
    parser.add_argument("--mixup_pct", type=float, default=0.33)
    parser.add_argument("--blurry_recon", type=lambda x: x.lower() == 'true', default=True)

    # Multi-subject fMRI averaging strategy
    parser.add_argument("--fmri_average_mode", type=str, default="loss",
                        choices=["embedding", "loss"],
                        help='Where to average multi-subject fMRI: "embedding" (before loss) or "loss" (after loss)')
    parser.add_argument("--subject_loss_weights", type=str, default=None,
                        help='Per-subject loss weights (format: "sub-01:0.4,sub-02:0.2,..."). Only used when fmri_average_mode="loss"')

    # MindEye2 checkpoint for generation modules
    parser.add_argument('--mindeye2_ckpt_path', type=str, default=None, help='Path to MindEYE2 checkpoint for loading generation module weights')

    # Checkpoints and paths
    parser.add_argument('--model_dir', type=str, default=None, help='Full E2E checkpoint')
    parser.add_argument('--align_model_dir', type=str, default=None, help='Pretrained alignment checkpoint')
    parser.add_argument('--script_path', type=str, default=None)
    parser.add_argument('--datasets_dir', type=str, default=None)
    parser.add_argument('--images_df_dir', type=str, default='datasets/processed')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--emb_source', type=str, default='things', choices=['nsd', 'things'],
                        help='Which image source to use for generation targets (default: things)')

    return parser.parse_args()


def train(model: EEG_fMRI_E2E,
          data_loader: typing.Dict[str, torch.utils.data.DataLoader],
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          device: torch.device,
          num_epochs: int,
          ckpt_dir: str,
          clip_value: float = 1.0,
          mixup_pct: float = 0.33,
          use_amp: bool = True,
          logger: SummaryWriter = None,
          ckpt_interval: int = None,
          accelerator: Accelerator = None,
          local_rank: int = 0,):

    best_val_loss = float('inf')

    # Alignment Enable FLAG
    # Get one sample from training set and check if nsd_data_list is empty.
    ALIGNMENT_ENABLED = len(next(iter(data_loader['train']))[1]) > 0  # nsd_data_list is at index 1

    for epoch in range(num_epochs):
        model.train()
        use_mixco = (epoch < int(mixup_pct * num_epochs))

        # Accumulate losses
        accum = {k: [] for k in ['total', 'align', 'prior', 'clip', 'blur', 'mse', 'infonce', 'proto']}

        for batch in tqdm(data_loader['train'], desc=f"Epoch {epoch+1}/{num_epochs}", disable=(local_rank != 0)):
            (EEG, nsd_data_list, label, _, _, _, _, _,
             clip_target, vae_latents, cnx_features, cnx_blurry_features) = batch

            EEG = EEG.to(device)
            for nd in nsd_data_list:
                nd['fmri'] = nd['fmri'].to(device)
            label = label.to(device)
            clip_target = clip_target.to(device)
            vae_latents = vae_latents.to(device)
            cnx_features = cnx_features.to(device)
            cnx_blurry_features = cnx_blurry_features.to(device)

            # Stage 1: EEG encoder (BEFORE MixCo — raw EEG is never augmented)
            with torch.amp.autocast('cuda', enabled=use_amp):
                eeg_embeds = model.forward_encoder(EEG)  # (B, 4096)

            # Keep original embeddings for alignment loss
            eeg_embeds_orig = eeg_embeds.detach().clone()

            # MixCo on 4096-dim EMBEDDINGS only (safe — no temporal dynamics destroyed)
            perm, betas, select = None, None, None
            if use_mixco:
                eeg_embeds_mixed, perm, betas, select = mixco(
                    eeg_embeds.clone(), beta=0.15, s_thresh=0.5
                )
                # Also mix CLIP targets correspondingly
                clip_target_flat = clip_target.flatten(1)
                clip_target_flat = mixco_clip_target(clip_target_flat.clone(), perm, select, betas)
                clip_target_mixed = clip_target_flat.view(-1, 256, 1664)
            else:
                eeg_embeds_mixed = eeg_embeds
                clip_target_mixed = clip_target

            # Stage 2: BrainNetwork + loss (from mixed embeddings)
            with torch.amp.autocast('cuda', enabled=use_amp):
                gen_outputs = model.forward_generation(eeg_embeds_mixed)
                losses = model.calc_e2e_loss(
                    eeg_embeds_orig, nsd_data_list, label, gen_outputs,
                    clip_target_mixed, vae_latents, cnx_features, cnx_blurry_features,
                    epoch, num_epochs, perm, betas, select,
                )

            # Backward
            optimizer.zero_grad()
            accelerator.backward(losses['total'])
            if clip_value > 0:
                accelerator.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
            scheduler.step()

            for k in accum:
                accum[k].append(losses[k].item())

        # End of epoch — average training losses
        avg_train = {k: sum(v) / len(v) for k, v in accum.items()}
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Total: {avg_train['total']:.4f} "
              f"(Align: {avg_train['align']:.4f}, Prior: {avg_train['prior']:.4f}, "
              f"CLIP: {avg_train['clip']:.4f}, Blur: {avg_train['blur']:.4f})")

        # Validation (rank 0 only)
        if local_rank == 0:
            model.eval()
            val_accum = {k: [] for k in accum}
            outputs_list = []
            fmri_list = []
            with torch.no_grad():
                for batch in tqdm(data_loader['val'], desc = f"Val {epoch+1}"):
                    (EEG, nsd_data_list, label, _, _, _, _, _,
                     clip_target, vae_latents, cnx_features, cnx_blurry_features) = batch

                    # Move data to device
                    EEG = EEG.to(device)
                    # NOTE: nsd data for alignment loss is optional
                    for nd in nsd_data_list:
                        nd['fmri'] = nd['fmri'].to(device)
                    label = label.to(device)
                    clip_target = clip_target.to(device)
                    vae_latents = vae_latents.to(device)
                    cnx_features = cnx_features.to(device)
                    cnx_blurry_features = cnx_blurry_features.to(device)

                    # if ALIGNMENT_ENABLED, we take mean of all fmri data and
                    # append it to fmri_list for loss computation.
                    if ALIGNMENT_ENABLED:
                        # Take mean of all fmri data
                        fmri_mean = torch.stack([nd['fmri'] for nd in nsd_data_list]).mean(dim = 0)
                        # Append to fmri_list
                        fmri_list.append(fmri_mean)

                    with torch.amp.autocast('cuda', enabled=use_amp):
                        eeg_embeds = model.forward_encoder(EEG)
                        # if ALIGNMENT_ENABLED, we also append the eeg_embeds to
                        # outputs_list for loss computation.
                        if ALIGNMENT_ENABLED:
                            outputs_list.append(eeg_embeds)
                        gen_outputs = model.forward_generation(eeg_embeds)
                        losses = model.calc_e2e_loss(
                            eeg_embeds, nsd_data_list, label, gen_outputs,
                            clip_target, vae_latents, cnx_features, cnx_blurry_features,
                            epoch, num_epochs,
                        )

                    for k in val_accum:
                        val_accum[k].append(losses[k].item())

                # if ALIGNMENT_ENABLED, we compute alignment metrics,
                # else, wwe set them all 0
                if not ALIGNMENT_ENABLED:
                    mse, cos_sim, ret_acc_top1, ret_acc_top10 = 0.0, 0.0, 0.0, 0.0
                else:
                    all_outputs = torch.cat(outputs_list, dim = 0)
                    all_fmri = torch.cat(fmri_list, dim = 0)
                    mse, cos_sim, ret_acc_top1, ret_acc_top10 = \
                        model.align_model.get_metrics_for_alignment(all_outputs.squeeze(), all_fmri.squeeze())

            avg_val = {k: sum(v) / len(v) for k, v in val_accum.items()}

            print(f"Epoch [{epoch+1}/{num_epochs}] - Val Total: {avg_val['total']:.4f} "
                  f"(Align: {avg_val['align']:.4f}, Prior: {avg_val['prior']:.4f}, "
                  f"CLIP: {avg_val['clip']:.4f}, Blur: {avg_val['blur']:.4f})")
            print(f"  Metrics — MSE: {mse:.4f}, CosSim: {cos_sim:.4f}, "
                  f"Top1: {ret_acc_top1:.4f}, Top10: {ret_acc_top10:.4f}")

            # Checkpointing
            unwrapped_model = accelerator.unwrap_model(model)
            if ckpt_interval is not None and (epoch + 1) % ckpt_interval == 0:
                 checkpoint_path = f"{ckpt_dir}/model_epoch_{epoch+1}.pth"
                 unwrapped_model.save_model(checkpoint_path)
                 print(f"Saved checkpoint to {checkpoint_path}")
            else:
                if avg_val['total'] < best_val_loss:
                    best_val_loss = avg_val['total']
                    # Only delete the ones started with "best_model" to
                    # avoid accidentally deleting other checkpoints if ckpt_interval is also used
                    for filename in os.listdir(ckpt_dir):
                        if filename.startswith("best_model_epoch_"):
                            file_path = os.path.join(ckpt_dir, filename)
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                    checkpoint_path = f"{ckpt_dir}/best_model_epoch_{epoch+1}.pth"
                    unwrapped_model.save_model(checkpoint_path)
                    print(f"Saved best model to {checkpoint_path}")

            # ── TensorBoard logging ──
            if logger is not None:
                for k in avg_train:
                    logger.add_scalar(f'Loss_{k}/Train', avg_train[k], epoch)
                    logger.add_scalar(f'Loss_{k}/Val', avg_val[k], epoch)
                logger.add_scalar('Metrics/MSE', mse, epoch)
                logger.add_scalar('Metrics/Cosine_Similarity', cos_sim, epoch)
                logger.add_scalar('Metrics/Retrieval_Accuracy_Top1', ret_acc_top1, epoch)
                logger.add_scalar('Metrics/Retrieval_Accuracy_Top10', ret_acc_top10, epoch)
                optim_state = optimizer.state_dict()
                for i, group in enumerate(optim_state['param_groups']):
                    logger.add_scalar(f'Learning_Rate/Group_{i}', group['lr'], epoch)

        # Synchronize all processes at end of epoch
        accelerator.wait_for_everyone()


# Main function
if __name__ == "__main__":

    args = get_args()

    # Multi-GPU setup with accelerate
    local_rank = os.getenv('RANK')
    if local_rank is None:
        local_rank = 0
    else:
        local_rank = int(local_rank)

    num_devices = torch.cuda.device_count()
    if num_devices == 0: num_devices = 1

    accelerator = Accelerator(split_batches=False, mixed_precision="fp16" if args.use_amp else "no")
    device = accelerator.device
    print = accelerator.print  # only print on rank 0

    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    # Batch size scaling for multi-GPU
    global_batch_size = args.batch_size
    args.batch_size = global_batch_size // num_devices
    print(f"Global batch size: {global_batch_size}, per-GPU batch size: {args.batch_size}, num_devices: {num_devices}")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    print(f"Using device: {device}")

    # Create experiment folder (compute paths on all ranks, create dirs on main only)
    if args.experiment_folder is not None:
        log_dir = f"runs/{args.experiment_folder}/{args.experiment_name}_{int(time())}"
    else:
        log_dir = f"runs/{args.experiment_name}_{int(time())}"
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir = f"{log_dir}/checkpoints"

    logger = None
    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(tensorboard_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Logging to {log_dir}")

        if args.script_path and os.path.isfile(args.script_path):
            shutil.copy(args.script_path, f"{log_dir}/run_script.sh")
            print(f"Saved launch script to {log_dir}/run_script.sh")

        logger = SummaryWriter(log_dir=tensorboard_dir)

    # Load data (with pre-computed generation targets)
    data_loader = get_generation_data_loader(
        datasets_dir=args.datasets_dir,
        images_df_dir=args.images_df_dir,
        batch_size=args.batch_size,
        normalize_fmri=args.normalize_fmri,
        load_images=False,
        num_workers=args.num_workers,
        emb_source=args.emb_source,
    )

    # Build model config
    if args.backbone == 'CBraMod':
        encoder_config = {
            'encoder_type': 'CBraMod',
            'cuda': args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
            'pooling_type': args.pooling_type,
            'attention_heads': args.attention_heads,
            'num_tokens': args.num_tokens,
            'num_transformer_layers': args.num_transformer_layers,
            'num_attention_heads': args.num_attention_heads,
            'mlp_layers': args.mlp_layers,
            'embedding_dim': args.embedding_dim,
        }
    elif args.backbone == 'ATMS':
        encoder_config = {
            'encoder_type': 'ATMS',
            'cuda': args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir': args.foundation_dir,
            'num_channels': 63,
            'seq_len': 250,
            'emb_size': args.atms_emb_size,
            'proj_dim': 1024,
            'drop_proj': args.atms_drop_proj,
            'd_model': args.atms_d_model,
            'n_heads': args.atms_n_heads,
            'e_layers': 1,
            'd_ff': args.atms_d_ff,
            'dropout': args.atms_dropout,
            'factor': args.atms_factor,
            'out_mlp_dim': args.out_mlp_dim,
        }
    else:
        raise ValueError(f"Unsupported backbone type: {args.backbone}")

    # Parse and validate subject weights (front-load validation)
    subject_loss_weights = parse_and_validate_subject_weights(args.subject_loss_weights)

    model_config = {
        'EEG_Encoder': encoder_config,
        'Loss': {
            'mse_scale': args.mse_scale,
            'infonce_scale': args.infonce_scale,
            'proto_distill_scale': args.proto_distill_scale,
            'temperature': args.temperature,
            'normalize_fmri': args.normalize_fmri,
        },
        'Generation': {
            'n_blocks': 4,
            'drop': 0.15,
            'clip_size': 1664,
            'blurry_recon': args.blurry_recon,
            'clip_scale': 1.0,
            'prior_scale': args.prior_scale,
            'clip_loss_scale': args.clip_loss_scale,
            'blur_scale': args.blur_scale,
            'align_scale': args.align_scale,
            'mindeye2_ckpt_path': args.mindeye2_ckpt_path,
            'fmri_average_mode': args.fmri_average_mode,
            'subject_loss_weights': subject_loss_weights,
        },
    }

    model = EEG_fMRI_E2E(model_config).to(device)

    # Load alignment checkpoint if specified (warm-start encoder + projection)
    if args.align_model_dir is not None:
        model.load_alignment_checkpoint(args.align_model_dir, device)

    # Load full E2E checkpoint if specified
    if args.model_dir is not None:
        model.load_model(args.model_dir, device)
        print(f"Loaded full E2E model from {args.model_dir}")

    # Freeze EEG encoder if specified
    if args.freeze_encoder:
        for param in model.align_model.eeg_encoder.parameters():
            param.requires_grad = False
        print("Frozen EEG encoder parameters")

    # Freeze MindEye2 generation modules if specified
    if args.freeze_mindeye2:
        for param in model.brain_network.parameters():
            param.requires_grad = False
        for param in model.diffusion_prior.parameters():
            param.requires_grad = False
        print("Frozen MindEye2 generation module parameters")

    # ── Optimizer setup (MindEye2-style weight decay separation) ──
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    max_lr = args.lr

    # Separate parameters into groups
    encoder_decay, encoder_no_decay = [], []
    align_head_decay, align_head_no_decay = [], []
    brain_net_decay, brain_net_no_decay = [], []
    prior_decay, prior_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = any(nd in name for nd in no_decay)
        if name.startswith('align_model.eeg_encoder.'):
            (encoder_no_decay if is_no_decay else encoder_decay).append(param)
        elif name.startswith('align_model.'):
            (align_head_no_decay if is_no_decay else align_head_decay).append(param)
        elif name.startswith('brain_network.'):
            (brain_net_no_decay if is_no_decay else brain_net_decay).append(param)
        elif name.startswith('diffusion_prior.'):
            (prior_no_decay if is_no_decay else prior_decay).append(param)

    param_groups = [
        {'params': encoder_decay, 'lr': max_lr * 0.2, 'weight_decay': args.weight_decay},
        {'params': encoder_no_decay, 'lr': max_lr * 0.2, 'weight_decay': 0.0},
        {'params': align_head_decay, 'lr': max_lr, 'weight_decay': args.weight_decay},
        {'params': align_head_no_decay, 'lr': max_lr, 'weight_decay': 0.0},
        {'params': brain_net_decay, 'lr': max_lr, 'weight_decay': args.weight_decay},
        {'params': brain_net_no_decay, 'lr': max_lr, 'weight_decay': 0.0},
        {'params': prior_decay, 'lr': max_lr, 'weight_decay': args.weight_decay},
        {'params': prior_no_decay, 'lr': max_lr, 'weight_decay': 0.0},
    ]
    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]

    optimizer = optim.AdamW(param_groups)

    # Prepare model, optimizer, and train dataloader with accelerate
    model, optimizer = accelerator.prepare(model, optimizer)
    data_loader['train'] = accelerator.prepare(data_loader['train'])
    # Leave val/test dataloaders unprepared — only run on rank 0

    total_steps = args.epochs * len(data_loader['train'])
    max_lrs = [g['lr'] for g in param_groups]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        total_steps=total_steps,
        pct_start=min(2 / args.epochs, 0.1),
        final_div_factor=1000,
    )
    scheduler = accelerator.prepare(scheduler)

    print(f"Optimizer groups: {len(param_groups)}")
    for i, g in enumerate(param_groups):
        print(f"  Group {i}: {len(g['params'])} params, lr={g['lr']:.6f}, wd={g['weight_decay']}")

    # Train
    train(model, data_loader, optimizer, scheduler, device,
          args.epochs, ckpt_dir, args.clip_value, args.mixup_pct, args.use_amp, logger, args.ckpt_interval,
          accelerator, local_rank)

    if accelerator.is_main_process and logger is not None:
        logger.close()
    print("Training complete. Tensorboard logs saved to:", tensorboard_dir)
