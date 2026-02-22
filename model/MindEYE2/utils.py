# Utility functions from MindEyeV2
# Reference: 
# https://github.com/MedARC-AI/MindEyeV2/blob/main/src/utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple


def batchwise_cosine_similarity(Z: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    This function computes the batchwise cosine similarity between two tensors Z and B.
    Z is expected to be of shape (batch_size, feature_dim) and B is expected to be of shape (feature_dim, batch_size).
    The output will be a tensor of shape (batch_size, batch_size) where each element (i, j) represents the cosine similarity between the i-th row of Z and the j-th column of B.
    """
    Z = Z.flatten(1)
    B = B.flatten(1).T
    Z_norm = torch.linalg.norm(Z, dim = 1, keepdim = True)
    B_norm = torch.linalg.norm(B, dim = 0, keepdim = True)
    cosine_similarity = ((Z @ B) / (Z_norm @ B_norm)).T
    return cosine_similarity


def soft_clip_loss(preds: torch.Tensor, targs: torch.Tensor, temp: float = 0.125) -> torch.Tensor:
    """
    This function computes the soft CLIP loss between the predicted embeddings (preds) and the target embeddings (targs).
    The loss is calculated using the cosine similarity between the predicted and target embeddings, scaled by a temperature parameter (temp).
    The loss is computed in a bidirectional manner, meaning it considers both the similarity of preds to targs and the similarity of targs to preds, and averages the two losses.
    """
    clip_clip = (targs @ targs.T) / temp
    brain_clip = (preds @ targs.T) / temp
    loss1 = -(brain_clip.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss2 = -(brain_clip.T.log_softmax(-1) * clip_clip.softmax(-1)).sum(-1).mean()
    loss = (loss1 + loss2) / 2
    return loss


def soft_cont_loss(student_preds: torch.Tensor, teacher_preds: torch.Tensor, teacher_aug_preds: torch.Tensor, temp: float = 0.125) -> torch.Tensor:
    """
    This function computes the soft contrastive loss between the student predictions and the teacher predictions, as well as the teacher augmented predictions.
    The loss is calculated using the cosine similarity between the student predictions and the teacher augmented predictions, as well as the cosine similarity between the teacher predictions and the teacher augmented predictions, all scaled by a temperature parameter (temp).
    The loss is computed in a bidirectional manner, meaning it considers both the similarity of student_preds to teacher_aug_preds and the similarity of teacher_aug_preds to student_preds, as well as the similarity of teacher_preds to teacher_aug_preds and the similarity of teacher_aug_preds to teacher_preds, and averages the two losses.
    """
    teacher_teacher_aug = (teacher_preds @ teacher_aug_preds.T) / temp
    teacher_teacher_aug_t = (teacher_aug_preds @ teacher_preds.T) / temp
    student_teacher_aug = (student_preds @ teacher_aug_preds.T) / temp
    student_teacher_aug_t = (teacher_aug_preds @ student_preds.T) / temp

    loss1 = -(student_teacher_aug.log_softmax(-1) * teacher_teacher_aug.softmax(-1)).sum(-1).mean()
    loss2 = -(student_teacher_aug_t.log_softmax(-1) * teacher_teacher_aug_t.softmax(-1)).sum(-1).mean()
    loss = (loss1 + loss2) / 2
    return loss


def mixco(voxels, beta: float = 0.15, s_thresh: float = 0.5, perm: torch.Tensor = None, betas: torch.Tensor = None, select: torch.Tensor = None) \
    -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    This function performs the MixCo data augmentation technique on the input voxel data.
    It takes in the voxel data and applies a mixup-like augmentation by randomly permuting the voxels, sampling beta values from a Beta distribution, and selecting which voxels to mix based on a random threshold.
    The function returns the augmented voxels, the permutation used, the beta values used for mixing, and the selection mask indicating which voxels were mixed.
    """
    if perm is None:
        perm = torch.randperm(voxels.shape[0])
    voxels_shuffle = voxels[perm].to(voxels.device, dtype = voxels.dtype)
    if betas is None:
        betas = torch.distributions.Beta(beta, beta).sample([voxels.shape[0]]).to(
            voxels.device, dtype = voxels.dtype
        )
    if select is None:
        select = (torch.rand(voxels.shape[0]) <= s_thresh).to(voxels.device)
    betas_shape = [-1] + [1] * (len(voxels.shape) - 1)
    voxels[select] = (
        voxels[select] * betas[select].reshape(*betas_shape)
        + voxels_shuffle[select] * (1 - betas[select]).reshape(*betas_shape)
    )
    betas[~select] = 1
    return voxels, perm, betas, select


def mixco_clip_target(clip_target: torch.Tensor, perm: torch.Tensor, select: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    """
    This function applies the MixCo augmentation to the CLIP target embeddings based on the permutation, selection mask, and beta values used for the voxel data augmentation.
    It takes in the CLIP target embeddings and modifies them according to the same permutation and mixing strategy used for the voxel data, ensuring that the CLIP targets are consistent with the augmented voxel data.
    """
    clip_target_shuffle = clip_target[perm]
    clip_target[select] = (
        clip_target[select] * betas[select].reshape(-1, 1)
        + clip_target_shuffle[select] * (1 - betas[select]).reshape(-1, 1)
    )
    return clip_target


def mixco_nce(
    preds: torch.Tensor,
    targs: torch.Tensor,
    temp: float = 0.1,
    perm: torch.Tensor = None,
    betas: torch.Tensor = None,
    select: torch.Tensor = None,
    bidirectional: bool = True,
) -> torch.Tensor:
    
    """
    This function computes the MixCo Noise Contrastive Estimation (NCE) loss between the predicted embeddings (preds) and the target embeddings (targs), taking into account the MixCo augmentation applied to the voxel data and the CLIP targets.
    The loss is calculated using the cosine similarity between the predicted and target embeddings, scaled by a temperature parameter (temp), and adjusted based on the permutation, selection mask, and beta values used for the MixCo augmentation.
    The loss is computed in a bidirectional manner, meaning it considers both the similarity of preds to targs and the similarity of targs to preds, and averages the two losses if bidirectional is set to True. If the permutation, selection mask, and beta values are not provided, it defaults to a standard cross-entropy loss without the MixCo adjustments.
    """

    brain_clip = (preds @ targs.T) / temp

    if perm is not None and betas is not None and select is not None:
        probs = torch.diag(betas)
        probs[torch.arange(preds.shape[0]).to(preds.device), perm] = 1 - betas

        loss = -(brain_clip.log_softmax(-1) * probs).sum(-1).mean()
        if bidirectional:
            loss2 = -(brain_clip.T.log_softmax(-1) * probs.T).sum(-1).mean()
            loss = (loss + loss2) / 2
        return loss
    else:
        loss = F.cross_entropy(
            brain_clip, torch.arange(brain_clip.shape[0]).to(brain_clip.device)
        )
        if bidirectional:
            loss2 = F.cross_entropy(
                brain_clip.T,
                torch.arange(brain_clip.shape[0]).to(brain_clip.device),
            )
            loss = (loss + loss2) / 2
        return loss


def cosine_anneal(start: float, end: float, steps: int) -> torch.Tensor:
    """
    This function generates a cosine annealing schedule for a given start value, end value, and number of steps.
    The schedule is calculated using the cosine function to create a smooth transition from the start value to the end value over the specified number of steps, which can be useful for learning rate scheduling or other hyperparameter adjustments during training.
    """
    return end + (start - end) / 2 * (
        1 + torch.cos(torch.pi * torch.arange(steps) / (steps - 1))
    )