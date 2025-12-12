import torch
import numpy as np
from pathlib import Path
import json
import argparse
from enum import Enum
import logging
from tqdm import tqdm
from utils import *
from model.backbones import *
from model.architecture import *
from model.trainer import *
import wandb


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



# Add cascade weight info to wandb config

# ============================================================================
# Feature Processing
# ============================================================================

# TODO maybe delete 
class FeatureProcessor:
    """Handles Labram feature extraction and caching."""
    
    def __init__(self, labram_model: nn.Module, config: ModelConfig, device: str = 'cuda'):
        self.labram = labram_model
        self.config = config
        self.device = device
    
    def get_features_path(self, dataset_name: str) -> Path:
        """Get path for cached features."""
        return Path('/home/burger/canWeReally/data/processed_data') / f'{dataset_name}_labram_features.pt'
    
    def extract_and_save_features(self, data_dict: Dict[str, torch.Tensor], 
                                dataset_name: str) -> Dict[str, torch.Tensor]:
        """Extract Labram features and save them."""
        print("Extracting Labram features...")
        features_path = self.get_features_path(dataset_name)
        features_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract features
        with torch.no_grad():
            eeg_data = data_dict['data'].to(self.device)
            features = []
            
            # Process in batches to avoid OOM
            batch_size = 128
            for i in range(0, len(eeg_data), batch_size):
                batch = eeg_data[i:i + batch_size]
                batch_features = self.labram.forward_features(batch)
                features.append(batch_features.cpu())
                
            features = torch.cat(features, dim=0)
        
        # Save features and metadata
        feature_dict = {
            'features': features,
            'subjects': data_dict['subjects'],
            'tasks': data_dict['tasks'],
            'runs': data_dict['runs'],
            'data_mean': features.mean(dim=0),
            'data_std': features.std(dim=0)
        }
        
        torch.save(feature_dict, features_path)
        print(f"Saved features to {features_path}")
        return feature_dict
    
    def load_or_extract_features(self, data_dict: Dict[str, torch.Tensor], 
                               dataset_name: str) -> Dict[str, torch.Tensor]:
        """Load cached features or extract if not available."""
        features_path = self.get_features_path(dataset_name)
        
        if features_path.exists():
            print(f"Loading pre-computed features from {features_path}")
            return torch.load(features_path)
        
        return self.extract_and_save_features(data_dict, dataset_name)





# ============================================================================
# Main Training Script
# ============================================================================


def parse_training_args():
    """Enhanced argument parser with clear mode selection"""
    parser = argparse.ArgumentParser(description='Modular DVAE Training')
    
    # Data arguments
    parser.add_argument('--data-file', required=True,
                       help='Path to .pt file (raw data or pre-extracted features)')
    parser.add_argument('--dataset', choices=['erps', 'sleepedfx', 'MI_eeg'],
                       help='Dataset type (for shape inference)')
    
    # Training mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--mode-a-raw-learnable', action='store_true',
                           help='Mode A: Raw data ->  Learnable Projector -> DVAE')
    mode_group.add_argument('--mode-b-frozen-backbone', action='store_true',
                           help='Mode B: Frozen Backbone ->  DVAE')
    mode_group.add_argument('--mode-c-two-stage', action='store_true',
                           help='Mode C: Stage1(Finetune Backbone on T_A) ->  Stage2(DVAE on T_V)')
    mode_group.add_argument('--mode-d-joint-frozen', action='store_true',
                           help='Mode D: Joint Training ->  Freeze Backbone ->  Continue DVAE')
    mode_group.add_argument('--mode-e-signal-features', action='store_true',
                           help='Mode E: Pre-extracted Signal Features -> DVAE')
    mode_group.add_argument('--mode-f-classifier-only', action='store_true',
                           help='Mode F: Frozen backbone -> Classifier Only')
    
    # Backbone arguments (required for modes B, C, D)
    parser.add_argument('--backbone', choices=['labram', 'cbramod'],
                       help='Backbone architecture (required for modes B/C/D)')
    parser.add_argument('--backbone-weights',
                       help='Path to backbone weights (required for modes B/C/D)')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Mode-specific arguments
    parser.add_argument('--stage1-epochs', type=int, default=10,
                       help='[Mode C] Epochs for backbone adaptation stage')
    parser.add_argument('--disjoint-split-ratio', type=float, default=0.3,
                       help='[Mode C] Ratio of train data for T_A')
    parser.add_argument('--freeze-after-epoch', type=int, default=20,
                       help='[Mode D] Epoch to freeze backbone')
    parser.add_argument('--include-subject-classifiers', action='store_true',
                       help='[Mode B/C] Whether to include subject classifiers during Stage 1 finetuning')
    
    # Projector architecture (Mode A)
    parser.add_argument('--projector-hidden-dim', type=int, default=512,
                       help='[Mode A] Hidden dimension for projector')
    parser.add_argument('--projector-dropout', type=float, default=0.1,
                       help='[Mode A] Dropout rate for projector')
    # Logging and saving
    parser.add_argument('--run-name', type=str, default='',
                       help='Name for the training run (in WandB)')

    parser.add_argument('--save-dir', required=True)
    
    return parser.parse_args()

def validate_args(args):
    """Validate argument combinations"""
    mode = get_training_mode(args)
    
    # Modes B, C, D require backbone specification
    if mode in [TrainingMode.FROZEN_BACKBONE, TrainingMode.TWO_STAGE_DISJOINT, 
                TrainingMode.JOINT_THEN_FROZEN, TrainingMode.CLASSIFIER_ONLY]:
        if not args.backbone:
            raise ValueError(f"{mode.value} requires --backbone")
        if not args.backbone_weights:
            raise ValueError(f"{mode.value} requires --backbone-weights")
    
    # Mode A and E shouldn't specify backbone
    if mode == TrainingMode.RAW_LEARNABLE: # or mode == TrainingMode.SIGNAL_FEATURES:
        if args.backbone or args.backbone_weights:
            logging.warning("Mode A and E ignores --backbone and --backbone-weights")
    
    # Validate file paths exist
    from pathlib import Path
    if not Path(args.data_file).exists():
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    if args.backbone_weights and not Path(args.backbone_weights).exists():
        raise FileNotFoundError(f"Backbone weights not found: {args.backbone_weights}")


def main():
    args = parse_training_args()
    validate_args(args)
    mode = get_training_mode(args)
    
    # Create sensate wandb_config using args
    wandb_config = vars(args)
    print("Initializing Weights & Biases logging...")
    print(f"wandb config: {wandb_config}")
    wandb_config['training_mode'] = mode.value
    wandb.init(project="dvae_disentanglement", config=wandb_config, group='', entity='giuseppe-facchi-phuselab')

    
    # ----------------------------------------------------
    # 1. SETUP DUAL-STREAM LOGGING (FILE AND CONSOLE)
    # ----------------------------------------------------
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(args.save_dir)
    logging.info(f"Training Mode: {mode.value}")
    logging.info(f"Arguments: {vars(args)}\n")
    
    # Load data
    logging.info(f"Loading data from {args.data_file}...")
    data_dict = torch.load(args.data_file, weights_only=False)
    # Sometimes data is stored under 'features', sometimes under 'data', someitmes 'X'
    if 'data' not in data_dict:
        if 'features' in data_dict:
            data_dict['data'] = data_dict.pop('features')
        elif 'X' in data_dict:
            data_dict['data'] = data_dict.pop('X')
        else:
            raise KeyError("Data file must contain 'data', 'features', or 'X' key")
    print(f"Shape of loaded data: {data_dict['data'].shape}")
    # data shape can be of size (N, C, T) or (N, F) depending on raw or features
    if len(data_dict['data'].shape) == 3:
        data_shape = (data_dict['data'].shape[1], data_dict['data'].shape[2])  # (C, T)
    elif len(data_dict['data'].shape) == 2:
        data_shape = (data_dict['data'].shape[0], data_dict['data'].shape[1], )  # (F,)
    logging.info(f"Data shape: {data_dict['data'].shape} -> Input: {data_shape}")
    
    
    # Create backbone
    backbone, feature_dim, _ = create_backbone(args, data_shape)
    
    LoaderClass, has_features = determine_loader_type(data_dict, args.dataset)
    
    # Create data loaders based on mode
    if mode == TrainingMode.TWO_STAGE_DISJOINT:
        train_loader_stage1, train_loader_stage2, val_loader, test_loader, custom_train_stage2 = create_disjoint_loaders(
            data_dict, args.dataset, args.batch_size, args.disjoint_split_ratio,
            num_samples_per_epoch=args.epochs * 1000
        )
        # Get class counts from stage 2 loader (the main training set)
        # We need to extract the custom loader from the DataLoader wrapper
        num_subjects = len(custom_train_stage2.unique_subjects)
        num_tasks = len(custom_train_stage2.unique_tasks)
    else:
        train_loader, val_loader, test_loader, custom_train = create_standard_loaders(
            data_dict, args.dataset, args.batch_size,
            num_samples_per_epoch=args.epochs * 1000
        )
        # Infer class counts from first batch
        num_subjects = len(custom_train.unique_subjects)
        num_tasks = len(custom_train.unique_tasks)
    
    # if PrecomputedFeatureLoader is used, we are loading features, so this means that we dont need to pass the data through the backbone
    if mode in [TrainingMode.FROZEN_BACKBONE, TrainingMode.TWO_STAGE_DISJOINT, TrainingMode.JOINT_THEN_FROZEN] and LoaderClass == PrecomputedFeatureLoader:
        # We are in a pre-trained/frozen mode AND loading preextracted features.
        # Sooo, the feature extraction step (backbone) has to be skipped.
        model_backbone = None
        # The feature_dim shouuuld already correct from create_backbone
        logging.info("Disabling feature_extractor in DisentangledEEGModel (using preextracted features).")
        
    elif mode == TrainingMode.SIGNAL_FEATURES:
        # We are loading pre-extracted features, so skip the backbone entirely
        model_backbone = None
        logging.info("Mode E selected: Using pre-extracted signal features, skipping backbone.")
    elif mode == TrainingMode.RAW_LEARNABLE and len(data_dict['data'].shape) == 2:
        # Safety check: If mode A is chosen but data is features, this might be a mistake.
        # For now, treat it like an empty feature extractor if raw data is not 3D.
        model_backbone = None
        logging.warning("Mode A selected, but input data is 2D features. Assuming data is features.")
    else:
        # This covers:
        # 1. Mode A (Raw Data -> Learnable Projector)
        # 2. Modes B/C/D (Raw Data -> Backbone) - *if the data file was raw EEG*
        # 3. Modes B/C/D (Features -> Backbone) - If the Backbone is only a projector (not typical, but safe)
        model_backbone = backbone
    
    # Clean up data_dict
    del data_dict
    
    # print trainable parameters 
    for name, param in model_backbone.named_parameters():
        if param.requires_grad:
            logging.info(f"BACKBONE TRAINABLE: {name} - {param.shape}")
        else:
            logging.info(f"BACKBONE FROZEN: {name} - {param.shape}")
    print("Finished printing backbone parameters")
    #exit() #! ACHTUNG LIA
    # Create config with inferred class counts
    config = ModelConfig(
        n_channels=data_shape[0],
        time_samples=data_shape[1],
        freeze_feature_extractor=(mode == TrainingMode.FROZEN_BACKBONE),
        # encoders={
        #     'subject': EncoderConfig('subject', enabled=True, latent_dim=32, 
        #                             num_classes=num_subjects),
        #     'task': EncoderConfig('task', enabled=True, latent_dim=32, 
        #                          num_classes=num_tasks),
        #     'noise': EncoderConfig('noise', enabled=True, latent_dim=32, 
        #                           num_classes=None),
        # },
        encoders={
            'task': EncoderConfig('task', enabled=True, latent_dim=32, 
                                 num_classes=num_tasks),
        },
        # loss_config=LossConfig(
        #     self_reconstruction=True,
        #     self_reconstruction_weight=1.0,
        #     kl_divergence=True,
        #     kl_weight=0.001,
        #     classification=True,
        #     classification_weight=1.0,
        #     latent_consistency=True,
        #     latent_consistency_weight=1.0,
        #     self_cycle=True, 
        #     self_cycle_weight=0.5,
        #     cross_subject_intra_class=False, # non funziona 
        #     cross_subject_intra_class_weight=0.3,
        #     cross_subject_cross_class=False, # non funziona 
        #     cross_subject_cross_class_weight=0.3,
        #     cross_class_cycle_feature=False,
        #     cross_class_cycle_feature_weight=0.3,
        #     cross_class_cycle_signal=False,
        #     cross_class_cycle_signal_weight=0.3,
        #     knowledge_distillation=True, # tut
        #     kd_weight=0.5,
        #     kd_temperature=3.0,
        #     adversarial=False, # DVAETrainer' object has no attribute 'optimizer_D'
        #     adversarial_weight=1.0,
        #     lambda_gp=10.0,
        #     use_stft_loss=True, # tut 
        #     adaptive_balancing=True # tut 
        # ),
        loss_config=LossConfig(
            self_reconstruction=True,
            self_reconstruction_weight=1.0,
            kl_divergence=True,
            kl_weight=0.001,
            classification=True,
            classification_weight=1.0,
            latent_consistency=False,
            latent_consistency_weight=1.0,
        ),
        generator_hidden_dims =(64, 128)
    )
    
    logging.info(f"Inferred {num_subjects} subjects, {num_tasks} tasks")
    
    # Save config
    save_dir = Path(args.save_dir)
    with open(save_dir / 'config.json', 'w') as f:
        config_dict = {
            'mode': mode.value,
            'n_channels': config.n_channels,
            'time_samples': config.time_samples,
            'freeze_feature_extractor': config.freeze_feature_extractor,
            'encoders': {k: vars(v) for k, v in config.encoders.items()},
            'loss_config': vars(config.loss_config),
        }
        json.dump(config_dict, f, indent=2)
    
    # add the config_dict to wandb config
    wandb.config.update(config_dict)
    if args.run_name != '':
        wandb.run.name = args.run_name
    else:
        wandb.run.name = args.save_dir.split('/')[-1]

    # Create model
    if mode == TrainingMode.CLASSIFIER_ONLY:
        logging.info("Mode F selected: Classifier Only on Frozen Backbone Features.")
        start_ft_classifier_only = False
        continue_ft_classifier_only = True
        if start_ft_classifier_only:
            model = SimpleFeaturesClassifier(model_backbone, feature_dim, config, num_classes=num_tasks)
        if continue_ft_classifier_only:
            # model_init = SimpleFeaturesClassifier(model_backbone, feature_dim, config, num_classes=num_tasks)
            # model = SimpleFeaturesClassifierContinued(model_init, previous_model_path=args.backbone_weights)
            model = SimpleFeaturesClassifier(model_backbone, feature_dim, config, num_classes=num_tasks)
            print("\n\n\n\nLoading backbone weights for Classifier Only model...")
            # load weights
            model.load_state_dict(torch.load(args.backbone_weights)['model_state_dict'])
            model.set_required_grad_for_backbone(requires_grad=True)
            # re-initialize classifier head
            model.classifier = MLPClassifier(input_dim=feature_dim, hidden_dims=[32], num_classes=num_tasks)



    else:
        model = DisentangledEEGModel(model_backbone, feature_dim, config)

    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    total_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters (trainable + frozen): {total_total_params:,}")
    logging.info(f"Total trainable parameters: {total_params:,}")

    
    # Create loss and optimizer
    loss_fn = DisentanglementLoss(config.loss_config)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01
    )
    
    # Execute training based on mode
    if mode in [TrainingMode.RAW_LEARNABLE, TrainingMode.FROZEN_BACKBONE, TrainingMode.SIGNAL_FEATURES, TrainingMode.CLASSIFIER_ONLY]:
        trainer = DVAETrainer(model, loss_fn, optimizer, mode=mode, save_dir=args.save_dir)
        if mode == TrainingMode.CLASSIFIER_ONLY:
            trainer.tmp_split_optimizer(train_backbone=False)
        trainer.train(train_loader, val_loader, args.epochs, wandb_run=wandb)
        final_trainer = trainer
    
    elif mode == TrainingMode.TWO_STAGE_DISJOINT:
        
        # --- STAGE 1 SETUP ---
        # Goal: Use the 4D output for external classification (CBraMod paper head)
        
        # 1. Set the backbone to output 4D (DISABLE pooling)
        model.set_feature_pooling(enable_pooling=False) 
        # 2. Start Stage 1 Trainer (uses the new 4D output)
        
        if args.include_subject_classifiers:
            logging.info("Including subject classifiers during Stage 1 finetuning.")
            stage1_trainer = BackboneFinetuneTainer(model, loss_fn, args.lr, use_external_classifier=True, include_subject_classifiers=True, num_classes_subject=num_subjects, num_classes_task=num_tasks)
        else:
            logging.info("Excluding subject classifiers during Stage 1 finetuning.")
            stage1_trainer = BackboneFinetuneTainer(model, loss_fn, args.lr, use_external_classifier=True, include_subject_classifiers=False, num_classes_subject=num_subjects, num_classes_task=num_tasks)
        stage1_trainer.train(train_loader_stage1, args.stage1_epochs, wandb_run=wandb)
        
        # save the backbone weights after stage 1 without the classification head
        logging.info("Saving backbone weights after Stage 1...")
        backbone_save_path = save_dir / 'backbone_after_stage1.pth'
        torch.save(model.feature_extractor.model.state_dict(), backbone_save_path)
        
        # --- STAGE 2 SETUP ---
        # Goal: Use the 2D output, now stable, for DVAE training
        
        # 1. Set the backbone to output 2D (ENABLE pooling)
        model.set_feature_pooling(enable_pooling=True) 
        
        # 2. Reactivate DVAE components
        model.set_required_grad_for_dvae(True)
        
        # print required grads for debugging
        for name, param in model.named_parameters():
            if param.requires_grad:
                logging.info(f"Stage 2 TRAINABLE: {name} - {param.shape}")
            else:
                logging.info(f"Stage 2 FROZEN: {name} - {param.shape}")
        
        
        # 2. Start Stage 2 Trainer (DVAE requires 2D input for its internal heads)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=0.01
        )
        stage2_trainer = DVAETrainer(model, loss_fn, optimizer, mode=mode, save_dir=args.save_dir)
        stage2_trainer.train(train_loader_stage2, val_loader, args.epochs, wandb_run=wandb)
        final_trainer = stage2_trainer
    
    elif mode == TrainingMode.JOINT_THEN_FROZEN:
        trainer = DVAETrainer(model, loss_fn, optimizer, mode=mode, save_dir=args.save_dir)
        trainer.train(train_loader, val_loader, args.epochs, 
                     freeze_at_epoch=args.freeze_after_epoch, wandb_run=wandb)
        final_trainer = trainer
    
    # Final evaluation
    logging.info("\n" + "="*60)
    logging.info("FINAL EVALUATION")
    logging.info("="*60 + "\n")
    if mode == TrainingMode.CLASSIFIER_ONLY:
        logging.info("Evaluating Classifier Only Model...")
        test_metrics = final_trainer.validate_classifier_only(test_loader)
    else:
        test_metrics = final_trainer.validate(test_loader)
    for k, v in test_metrics.items():
        logging.info(f"  {k}: {v:.4f}")
    
    # Save test metrics
    with open(save_dir / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
    
    """
    Example command to run the script:
    python disentangle_eeg.py \
    --features-file /home/burger/canWeReally/data/processed_data/heheheMI_eeg_cbramod_patch60_features.pt \
    --dataset MI_eeg \
    --save-dir experiments/mi_eeg_disentanglement_cbramod_features \
    --epochs 50 \
    --batch-size 32
    
    
    # Insert raw data in disentanglement module 
    python disentangle_eeg.py \
    --features-file /home/burger/canWeReally/data/processed_data/sleepedfx_cbramod_data.pt \
    --dataset sleepedfx \
    --save-dir experiments/sleep_disentanglement_raw \
    --epochs 50 \
    --batch-size 32
    
    """
