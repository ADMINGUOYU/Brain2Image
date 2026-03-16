# Training code for EEG-CLIP alignment
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import typing
from time import time
import os
import shutil

from model.EEG_CLIP_align import EEG_CLIP_Align
from data.EEG_CLIP_align_dataset import get_clip_align_data_loader


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EEG-CLIP alignment model")

    # General training settings
    parser.add_argument('--seed',              type=int,   default=648)
    parser.add_argument('--cuda',              type=int,   default=1)
    parser.add_argument('--epochs',            type=int,   default=100)
    parser.add_argument('--batch_size',        type=int,   default=64)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--multi_lr',          type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--backbone_lr_scale', type=float, default=0.2,
                        help='LR multiplier for eeg_encoder.backbone when multi_lr=True')
    parser.add_argument('--warmup_epochs',     type=int,   default=0,
                        help='Number of linear warmup epochs before cosine annealing')
    parser.add_argument('--weight_decay',      type=float, default=5e-2)
    parser.add_argument('--optimizer',         type=str,   default='AdamW')
    parser.add_argument('--clip_value',        type=float, default=1.0)
    parser.add_argument('--experiment_folder', type=str,   default=None)
    parser.add_argument('--experiment_name',   type=str,   default='EEG_CLIP_align')

    # Encoder backbone
    parser.add_argument('--backbone',                type=str, default='ATMS', choices=['CBraMod', 'ATMS'])
    parser.add_argument('--frozen',                  type=lambda x: x.lower() == 'true', default=False)
    parser.add_argument('--use_pretrained_weights',  type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--foundation_dir',          type=str, default=None)

    # ATMS-specific
    parser.add_argument('--atms_emb_size',  type=int,   default=40)
    parser.add_argument('--out_mlp_dim',    type=int,   default=0,
                        help='ATMS output MLP dim; 0 means None (no extra projection, outputs 1024 directly)')
    parser.add_argument('--atms_drop_proj', type=float, default=0.5)
    parser.add_argument('--atms_d_model',   type=int,   default=250)
    parser.add_argument('--atms_n_heads',   type=int,   default=4)
    parser.add_argument('--atms_d_ff',      type=int,   default=256)
    parser.add_argument('--atms_dropout',   type=float, default=0.25)
    parser.add_argument('--atms_factor',    type=int,   default=1)

    # CBraMod-specific
    parser.add_argument('--pooling_type',            type=str, default='flatten',
                        choices=['flatten', 'attention', 'multitoken_vit'])
    parser.add_argument('--embedding_dim',           type=int, default=1024)
    parser.add_argument('--mlp_layers',              type=int, default=2)
    parser.add_argument('--attention_heads',         type=int, default=8)
    parser.add_argument('--num_tokens',              type=int, default=4)
    parser.add_argument('--num_transformer_layers',  type=int, default=4)
    parser.add_argument('--num_attention_heads',     type=int, default=4)

    # Full checkpoint loading
    parser.add_argument('--model_dir',   type=str, default=None)
    parser.add_argument('--script_path', type=str, default=None)

    # Dataset
    parser.add_argument('--datasets_dir', type=str, default=None)
    parser.add_argument('--num_workers',  type=int, default=16)

    # Loss hyperparameters
    parser.add_argument('--mse_scale',     type=float, default=1.0)
    parser.add_argument('--infonce_scale', type=float, default=0.2)
    parser.add_argument('--normalize_clip',
                        type=lambda x: x.lower() == 'true', default=True)

    return parser.parse_args()


def train(model: EEG_CLIP_Align,
          data_loader: typing.Dict[str, torch.utils.data.DataLoader],
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler._LRScheduler,
          device: torch.device,
          num_epochs: int,
          ckpt_dir: str,
          clip_value: float = -1.0,
          logger: SummaryWriter = None):

    best_val_loss = float('inf')

    for epoch in range(num_epochs):

        model.train()
        total_losses   = []
        mse_losses     = []
        infonce_losses = []

        for EEG, clip_H_target, things_img_idx in tqdm(
                data_loader['train'], desc=f"Epoch {epoch+1}/{num_epochs}"):

            EEG          = EEG.to(device)
            clip_H_target = clip_H_target.to(device)
            things_img_idx = things_img_idx.to(device)

            optimizer.zero_grad()
            out = model.forward(EEG)
            total, mse, infonce = model.calc_alignment_loss(out, clip_H_target)
            total.backward()

            if clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

            optimizer.step()
            scheduler.step()

            total_losses.append(total.item())
            mse_losses.append(mse.item())
            infonce_losses.append(infonce.item())

        avg_train_loss   = sum(total_losses)   / len(total_losses)
        avg_mse_loss     = sum(mse_losses)     / len(mse_losses)
        avg_infonce_loss = sum(infonce_losses) / len(infonce_losses)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} "
              f"(MSE: {avg_mse_loss:.4f}, InfoNCE: {avg_infonce_loss:.4f})")

        # ── Validation ────────────────────────────────────────────────────────
        outputs      = []
        clip_targets = []
        val_loss = 0.0
        mse_val_loss = 0.0
        infonce_val_loss = 0.0

        with torch.no_grad():
            model.eval()
            for EEG, clip_H_target, things_img_idx in tqdm(
                    data_loader['val'], desc=f"Val Epoch {epoch+1}"):

                EEG           = EEG.to(device)
                clip_H_target = clip_H_target.to(device)
                things_img_idx = things_img_idx.to(device)

                clip_targets.append(clip_H_target)
                out = model.forward(EEG)
                outputs.append(out)

                total, mse, infonce = model.calc_alignment_loss(out, clip_H_target)
                val_loss         += total.item()
                mse_val_loss     += mse.item()
                infonce_val_loss += infonce.item()

            avg_val_loss         = val_loss         / len(data_loader['val'])
            avg_mse_val_loss     = mse_val_loss     / len(data_loader['val'])
            avg_infonce_val_loss = infonce_val_loss / len(data_loader['val'])

            outputs      = torch.cat(outputs,      dim=0)
            clip_targets = torch.cat(clip_targets, dim=0)
            mse_m, cos_sim, retrieval_acc, retrieval_acc_top10 = \
                model.get_metrics_for_alignment(outputs, clip_targets)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {avg_val_loss:.4f} "
              f"(MSE: {mse_m:.4f}, CosSim: {cos_sim:.4f}, "
              f"Top1: {retrieval_acc:.4f}, Top10: {retrieval_acc_top10:.4f})")

        # ── Checkpointing ─────────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            for filename in os.listdir(ckpt_dir):
                file_path = os.path.join(ckpt_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            ckpt_path = f"{ckpt_dir}/best_model_epoch_{epoch+1}.pth"
            model.save_model(ckpt_path)
            print(f"Saved best model to {ckpt_path}")

        # ── TensorBoard ───────────────────────────────────────────────────────
        if logger is not None:
            logger.add_scalar('Loss/Train',         avg_train_loss,      epoch)
            logger.add_scalar('Loss/Val',           avg_val_loss,        epoch)
            logger.add_scalar('MSE_Loss/Train',     avg_mse_loss,        epoch)
            logger.add_scalar('MSE_Loss/Val',       avg_mse_val_loss,    epoch)
            logger.add_scalar('InfoNCE_Loss/Train', avg_infonce_loss,    epoch)
            logger.add_scalar('InfoNCE_Loss/Val',   avg_infonce_val_loss, epoch)
            logger.add_scalar('Metrics/MSE',                    mse_m,               epoch)
            logger.add_scalar('Metrics/Cosine_Similarity',      cos_sim,             epoch)
            logger.add_scalar('Metrics/Retrieval_Accuracy_Top1', retrieval_acc,      epoch)
            logger.add_scalar('Metrics/Retrieval_Accuracy_Top10', retrieval_acc_top10, epoch)
            optim_state = optimizer.state_dict()
            for i, group in enumerate(optim_state['param_groups']):
                logger.add_scalar(f'Learning_Rate/Group_{i}', group['lr'], epoch)


def test(model: EEG_CLIP_Align,
         data_loader: torch.utils.data.DataLoader,
         device: torch.device,
         logger: SummaryWriter = None):

    model.eval()
    outputs      = []
    clip_targets = []

    with torch.no_grad():
        for EEG, clip_H_target, things_img_idx in tqdm(data_loader, desc="Testing"):
            EEG           = EEG.to(device)
            clip_H_target = clip_H_target.to(device)
            clip_targets.append(clip_H_target)
            out = model.forward(EEG)
            outputs.append(out)

        outputs      = torch.cat(outputs,      dim=0)
        clip_targets = torch.cat(clip_targets, dim=0)
        mse, cos_sim, retrieval_acc, retrieval_acc_top10 = \
            model.get_metrics_for_alignment(outputs, clip_targets)
        print(f"Test — MSE: {mse:.4f}, CosSim: {cos_sim:.4f}, "
              f"Top1: {retrieval_acc:.4f}, Top10: {retrieval_acc_top10:.4f}")

        if logger is not None:
            logger.add_scalar('Test/MSE',                    mse)
            logger.add_scalar('Test/Cosine_Similarity',      cos_sim)
            logger.add_scalar('Test/Retrieval_Accuracy_Top1', retrieval_acc)
            logger.add_scalar('Test/Retrieval_Accuracy_Top10', retrieval_acc_top10)


if __name__ == "__main__":

    args = get_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Logging / checkpoint dirs ─────────────────────────────────────────────
    if args.experiment_folder:
        log_dir = f"runs/{args.experiment_folder}/{args.experiment_name}_{int(time())}"
    else:
        log_dir = f"runs/{args.experiment_name}_{int(time())}"
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_dir = f"{log_dir}/tensorboard"
    ckpt_dir        = f"{log_dir}/checkpoints"
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(ckpt_dir,        exist_ok=True)
    print(f"Logging to {log_dir}")

    if args.script_path and os.path.isfile(args.script_path):
        shutil.copy(args.script_path, f"{log_dir}/run_script.sh")

    logger = SummaryWriter(log_dir=tensorboard_dir)

    # ── Data ─────────────────────────────────────────────────────────────────
    data_loader = get_clip_align_data_loader(
        args.datasets_dir,
        batch_size=args.batch_size,
        normalize_clip=args.normalize_clip,
        num_workers=args.num_workers,
    )

    # ── Model config ──────────────────────────────────────────────────────────
    if args.backbone == 'CBraMod':
        encoder_config = {
            'encoder_type':          'CBraMod',
            'cuda':                  args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir':        args.foundation_dir,
            'pooling_type':          args.pooling_type,
            'attention_heads':       args.attention_heads,
            'num_tokens':            args.num_tokens,
            'num_transformer_layers': args.num_transformer_layers,
            'num_attention_heads':   args.num_attention_heads,
            'mlp_layers':            args.mlp_layers,
            'embedding_dim':         args.embedding_dim,
        }
    elif args.backbone == 'ATMS':
        encoder_config = {
            'encoder_type':          'ATMS',
            'cuda':                  args.cuda,
            'use_pretrained_weights': args.use_pretrained_weights,
            'foundation_dir':        args.foundation_dir,
            'num_channels':          63,
            'seq_len':               250,
            'emb_size':              args.atms_emb_size,
            'proj_dim':              1024,
            'drop_proj':             args.atms_drop_proj,
            'd_model':               args.atms_d_model,
            'n_heads':               args.atms_n_heads,
            'e_layers':              1,
            'd_ff':                  args.atms_d_ff,
            'dropout':               args.atms_dropout,
            'factor':                args.atms_factor,
            'out_mlp_dim':           args.out_mlp_dim if args.out_mlp_dim > 0 else None,
        }
    else:
        raise ValueError(f"Unsupported backbone: {args.backbone}")

    model_config = {
        'EEG_Encoder': encoder_config,
        'Loss': {
            'mse_scale':      args.mse_scale,
            'infonce_scale':  args.infonce_scale,
            'normalize_clip': args.normalize_clip,
        },
    }

    model = EEG_CLIP_Align(model_config).to(device)

    if args.model_dir is not None:
        model.load_model(args.model_dir, device)
        print(f"Loaded model from {args.model_dir}")

    if args.frozen:
        for param in model.eeg_encoder.parameters():
            param.requires_grad = False
        print("Frozen EEG encoder parameters")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_class = getattr(optim, args.optimizer)
    backbone_params = []
    other_params    = []
    for name, param in model.named_parameters():
        if 'eeg_encoder.backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    print(f"Backbone params: {len(backbone_params)}, Other params: {len(other_params)}")

    if args.multi_lr:
        optimizer = optimizer_class([
            {'params': backbone_params, 'lr': args.lr * args.backbone_lr_scale},  # Pretrained backbone: lower LR
            {'params': other_params,    'lr': args.lr},                            # Projection head: full LR
        ], weight_decay=args.weight_decay)
    else:
        optimizer = optimizer_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps  = args.epochs * len(data_loader['train'])
    warmup_steps = args.warmup_epochs * len(data_loader['train'])
    cosine_steps = total_steps - warmup_steps

    if warmup_steps > 0:
        warmup_scheduler = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler  = CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler],
                                 milestones=[warmup_steps])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)

    # ── Train + Test ──────────────────────────────────────────────────────────
    train(model, data_loader, optimizer, scheduler, device,
          args.epochs, ckpt_dir, args.clip_value, logger)

    test(model, data_loader['test'], device, logger)

    logger.close()
    print("Training complete. TensorBoard logs:", tensorboard_dir)
