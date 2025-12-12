import torch
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
from data.dataloaders.sleep_loader import CustomLoaderSleep
from data.dataloaders.MI_loader import CustomLoaderMI
from data.dataloaders.precomputed_Feature_Loader import PrecomputedFeatureLoader
from data.dataloaders.erp_loader import CustomLoaderERP
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import logging
import numpy as np

from pathlib import Path

import sys


SEED = 42


class TrainingMode(Enum):
    """Clear enumeration of all training modes"""
    RAW_LEARNABLE = "raw_learnable"      # Mode A: Raw data, learnable projector
    FROZEN_BACKBONE = "frozen_backbone"  # Mode B: Frozen pretrained backbone
    TWO_STAGE_DISJOINT = "two_stage"     # Mode C: Separate datasets for stages
    JOINT_THEN_FROZEN = "joint_frozen"   # Mode D: Joint training then freeze
    SIGNAL_FEATURES = "signal_features" # Mode E: Pre-extracted features input
    CLASSIFIER_ONLY = "classifier_only"   # Mode F: Train only classifier on features


def get_training_mode(args) -> TrainingMode:
    """Determine which training mode is active"""
    if args.mode_a_raw_learnable:
        return TrainingMode.RAW_LEARNABLE
    elif args.mode_b_frozen_backbone:
        return TrainingMode.FROZEN_BACKBONE
    elif args.mode_c_two_stage:
        return TrainingMode.TWO_STAGE_DISJOINT
    elif args.mode_d_joint_frozen:
        return TrainingMode.JOINT_THEN_FROZEN
    elif args.mode_e_signal_features:
        return TrainingMode.SIGNAL_FEATURES
    elif args.mode_f_classifier_only:
        return TrainingMode.CLASSIFIER_ONLY
    else:
        raise ValueError("No training mode specified")

#===========================================================================#
# Helper functions
#===========================================================================
def segment_to_patches(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    # (B, C, T) -> (B, C, n_patches, patch_size)
    B, C, T = x.shape
    n_patches = T // patch_size
    if n_patches == 0:
        raise ValueError(f"patch_size {patch_size} larger than signal length {T}")
    x = x[:, :, : n_patches * patch_size]
    return x.view(B, C, n_patches, patch_size).contiguous()


def get_optimal_patch_size(signal_length: int, min_patches: int = 4, max_patches: int = 16) -> int:
    candidates = []
    for n_patches in range(min_patches, max_patches + 1):
        patch_size = signal_length // n_patches
        if signal_length % patch_size < 0.2 * patch_size:
            candidates.append((patch_size, n_patches))
    if not candidates:
        return signal_length // 8
    return min(candidates, key=lambda x: abs(x[1] - 8))[0]



def calculate_train_stats(data_tensor: torch.Tensor, train_indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates mean/std based ONLY on the data corresponding to the training indices."""
    
    # 1. Select the training subset data
    train_data = data_tensor[train_indices]
    
    # 2. Calculate mean and std across ALL dimensions (N, C, T) for the training subset
    mu = train_data.mean().detach().clone()
    std = train_data.std().detach().clone()
    
    return mu, std








# ============================================================================
# DataLoader Creation
# ============================================================================


def create_standard_loaders(data_dict: dict, dataset: str, batch_size: int, 
                           num_samples_per_epoch: int = 1000):
    """
    Create standard train/val/test loaders (Modes A, B, D).
    
    Args:
        data_dict: Loaded .pt file containing data, subjects, tasks, etc.
        dataset: Dataset name for determining splits
        batch_size: Batch size
        num_samples_per_epoch: Samples to draw per epoch (for iterative loaders)
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"Creating standard loaders for dataset: {dataset}")
    LoaderClass, has_features = determine_loader_type(data_dict, dataset)
    
    # If data has 'features' key, rename to 'data' for compatibility
    if 'features' in data_dict and 'data' not in data_dict:
        data_dict['data'] = data_dict.pop('features')
    
    # Calculate train statistics FIRST (before any splitting)
    unique_subjects = data_dict["subjects"].clone().unique().tolist()
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.3, random_state=SEED
    )
    val_subjects, test_subjects = train_test_split(
        test_subjects, test_size=0.5, random_state=SEED
    )
    
    # Calculate stats on training data only
    if 'data_mean' not in data_dict or 'data_std' not in data_dict:
        train_indices = [
            i for i, s in enumerate(data_dict["subjects"].tolist()) 
            if s in train_subjects
        ]
        data_dict['data_mean'], data_dict['data_std'] = calculate_train_stats(
            data_dict['data'].float(), train_indices
        )
        logging.info(f"Calculated train stats: mean={data_dict['data_mean'].item():.4f}, "
                    f"std={data_dict['data_std'].item():.4f}")
    
    # Create custom loader instances
    if LoaderClass == CustomLoaderERP:
        # ERPs dataset with predefined splits
        custom_train = CustomLoaderERP(data_dict, split='train')
        custom_val = CustomLoaderERP(data_dict, split='dev')
        custom_test = CustomLoaderERP(data_dict, split='test')
    elif LoaderClass == CustomLoaderMI:
        print("Using CustomLoaderMI for MI_eeg dataset")
        # MI_eeg dataset with predefined splits
        custom_train = CustomLoaderMI(data_dict, split='train')
        train_mean = custom_train.data_mean
        train_std = custom_train.data_std
        custom_val = CustomLoaderMI(data_dict, split='dev', data_mean=train_mean, data_std=train_std)
        custom_test = CustomLoaderMI(data_dict, split='test', data_mean=train_mean, data_std=train_std)
        print(f"Train size: {custom_train.size}, Val size: {custom_val.size}, Test size: {custom_test.size}")
    else:
        # Sleep or feature-based loaders
        custom_train = LoaderClass(data_dict, split_subjects=train_subjects, 
                                   split='train', location='cuda')
        custom_val = LoaderClass(data_dict, split_subjects=val_subjects, 
                                 split='eval', location='cuda')
        custom_test = LoaderClass(data_dict, split_subjects=test_subjects, 
                                  split='test', location='cuda')
    
    # Wrap in PyTorch DataLoaders
    train_loader = custom_train.get_dataloader(
        num_total_samples=num_samples_per_epoch,
        batch_size=batch_size,
        property='subject',
        random_sample=True
    )
    
    val_loader = custom_val.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False
    )
    
    test_loader = custom_test.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False
    )
    
    logging.info(f"Created standard loaders: "
                f"Train={custom_train.size}, Val={custom_val.size}, Test={custom_test.size}")
    
    return train_loader, val_loader, test_loader, custom_train


def create_disjoint_loaders(data_dict: dict, dataset: str, batch_size: int,
                           disjoint_ratio: float, num_samples_per_epoch: int = 1000):
    """
    Create disjoint train loaders for Mode C (two-stage training).
    
    Stage 1 (T_A): disjoint_ratio of training data for backbone finetuning
    Stage 2 (T_V): remaining training data for DVAE training
    
    Returns:
        train_loader_stage1, train_loader_stage2, val_loader, test_loader
    """
    LoaderClass, has_features = determine_loader_type(data_dict, dataset)
    
    if 'features' in data_dict and 'data' not in data_dict:
        data_dict['data'] = data_dict.pop('features')
    
    # Get unique subjects and create initial splits
    unique_subjects = data_dict["subjects"].clone().unique().tolist()
    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=0.3, random_state=SEED
    )
    val_subjects, test_subjects = train_test_split(
        test_subjects, test_size=0.5, random_state=SEED
    )
    
    # Calculate stats on FULL training set (important!)
    if 'data_mean' not in data_dict or 'data_std' not in data_dict:
        train_indices = [
            i for i, s in enumerate(data_dict["subjects"].tolist()) 
            if s in train_subjects
        ]
        data_dict['data_mean'], data_dict['data_std'] = calculate_train_stats(
            data_dict['data'].float(), train_indices
        )
    
    # Split training subjects into T_A and T_V
    t_a_subjects, t_v_subjects = train_test_split(
        train_subjects,
        test_size=(1.0 - disjoint_ratio),
        random_state=SEED
    )
    
    logging.info(f"Disjoint split: T_A={len(t_a_subjects)} subjects, "
                f"T_V={len(t_v_subjects)} subjects, "
                f"Val={len(val_subjects)} subjects, Test={len(test_subjects)} subjects")
    
    # Create custom loaders
    if LoaderClass == CustomLoaderERP:
        print("wuhuwhuhwuhwuhw")
        raise NotImplementedError("Mode C not implemented for ERPs dataset")
    
    custom_train_stage1 = LoaderClass(data_dict, split_subjects=t_a_subjects,
                                     split='train', location='cuda')
    custom_train_stage2 = LoaderClass(data_dict, split_subjects=t_v_subjects,
                                     split='train', location='cuda')
    # Compute train mean and std as mean of the statistics from stage1 and stage2
    full_train_mean = data_dict['data_mean']
    full_train_std = data_dict['data_std']
    
    custom_val = LoaderClass(data_dict, split_subjects=val_subjects,
                            split='eval', location='cuda', data_mean=full_train_mean, data_std=full_train_std)
    custom_test = LoaderClass(data_dict, split_subjects=test_subjects,
                             split='test', location='cuda', data_mean=full_train_mean, data_std=full_train_std)
    
    # Wrap in PyTorch DataLoaders
    train_loader_stage1 = custom_train_stage1.get_dataloader(
        num_total_samples=num_samples_per_epoch // 2,  # Smaller dataset
        batch_size=batch_size,
        property='subject',
        random_sample=True
    )
    
    train_loader_stage2 = custom_train_stage2.get_dataloader(
        num_total_samples=num_samples_per_epoch,
        batch_size=batch_size,
        property='subject',
        random_sample=True
    )
    
    val_loader = custom_val.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False
    )
    
    test_loader = custom_test.get_dataloader(
        num_total_samples=None,
        batch_size=batch_size,
        property=None,
        random_sample=False
    )
    
    logging.info(f"Disjoint loaders created: "
                f"T_A={custom_train_stage1.size}, T_V={custom_train_stage2.size}, "
                f"Val={custom_val.size}, Test={custom_test.size}")
    
    return train_loader_stage1, train_loader_stage2, val_loader, test_loader, custom_train_stage2


def determine_loader_type(data_dict: dict, dataset: str):
    """
    Determine which CustomLoader class to use based on data characteristics.
    
    Returns:
        loader_class: The appropriate CustomLoader class
        has_features: Whether data contains pre-extracted features
    """
    # Check if we have pre-extracted features
    has_features = 'features' in data_dict 
    
    if dataset == 'sleepedfx':
        return CustomLoaderSleep, has_features
    elif dataset == 'MI_eeg':
        return CustomLoaderMI, has_features
    elif dataset == 'erps':
        return CustomLoaderERP, has_features
    elif has_features:
        return PrecomputedFeatureLoader, True
    else: # we can delete sthis later 
        return CustomLoaderERP, False
    


def setup_logging(save_dir: str):
    """Setup dual-stream logging to file and console"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_file = save_path / 'training_log.txt'
    
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        force=True,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )



def infer_num_classes(custom_loader) -> Tuple[int, int]:
    """Extract number of unique subjects and tasks from a custom loader"""
    num_subjects = len(custom_loader.unique_subjects)
    num_tasks = len(custom_loader.unique_tasks)
    return num_subjects, num_tasks



def validate_label_ranges(train_loader, num_subjects: int, num_tasks: int):
    """
    CRITICAL: Verify that all labels in the dataset are within [0, num_classes-1].
    This catches label encoding bugs before they crash during training.
    """
    logging.info(f"\n{'='*60}")
    logging.info("VALIDATING LABEL RANGES")
    logging.info(f"{'='*60}")
    logging.info(f"Expected ranges: Subjects [0, {num_subjects-1}], Tasks [0, {num_tasks-1}]")
    
    all_subjects = []
    all_tasks = []
    
    # Sample multiple batches to get good coverage
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Check first 10 batches
            break
        all_subjects.append(batch[2])
        all_tasks.append(batch[3])
    
    all_subjects = torch.cat(all_subjects)
    all_tasks = torch.cat(all_tasks)
    
    subject_min, subject_max = all_subjects.min().item(), all_subjects.max().item()
    task_min, task_max = all_tasks.min().item(), all_tasks.max().item()
    
    logging.info(f"Observed ranges: Subjects [{subject_min}, {subject_max}], Tasks [{task_min}, {task_max}]")
    
    # Check for violations
    errors = []
    if subject_min < 0:
        errors.append(f"Subject labels have negative values: {subject_min}")
    if subject_max >= num_subjects:
        errors.append(f"Subject labels exceed num_classes: {subject_max} >= {num_subjects}")
    if task_min < 0:
        errors.append(f"Task labels have negative values: {task_min}")
    if task_max >= num_tasks:
        errors.append(f"Task labels exceed num_classes: {task_max} >= {num_tasks}")
    
    if errors:
        logging.error("LABEL RANGE VIOLATIONS DETECTED:")
        for error in errors:
            logging.error(f"  - {error}")
        raise ValueError("Label ranges do not match configured num_classes. "
                        "Check your CustomLoader label encoding logic!")
    
    logging.info("All labels within valid ranges")
    logging.info(f"{'='*60}\n")
