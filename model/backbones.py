import torch
import torch.nn as nn
from utils.helper_functions import *

from braindecode.models import Labram
from CBraMod_main.models.cbramod import CBraMod
from .architecture import *
import logging
from typing import Tuple, Optional, Dict

# import default dict
from collections import defaultdict
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from tqdm import tqdm



class FeatureBackbone(nn.Module):
    """Unified interface: forward(x[B,C,T]) -> features[B,F]."""
    feature_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError
    
    

# ===========================================================================#
# Feature Extractor Wrappers
# ===========================================================================#
# (LabramBackbone, CBraModBackbone classes follow here)
# This mirrors your extractor behavior: normalization, segment_to_patches, CBraMod .forward(patched), and mean over (C,S). I keep feature_dim=200 for both.
# -- LaBraM wrapper --
class LabramBackbone(FeatureBackbone):
    def __init__(self, n_chans: int, n_times: int, weights_path: str | None, patch_size: int, emb_size: int = 200):
        super().__init__()
        self.model = Labram(n_chans=n_chans, n_times=n_times, n_outputs=0, patch_size=patch_size, emb_size=emb_size)
        if weights_path:
            weights = torch.load(weights_path, map_location='cpu', weights_only=False)
            self.model.load_state_dict(weights['model'], strict=False)
        self.feature_dim = emb_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize per-channel per-trial like your extractor
        x = (x - x.mean(dim=2, keepdim=True)) / (x.std(dim=2, keepdim=True) + 1e-6)
        return self.model.forward_features(x)




# -- CBraMod wrapper (mirrors create_cbramod_model + extraction) --
class CBraModBackbone(FeatureBackbone):
    def __init__(self, in_dim: int, time_samples: int, patch_size: int, weights_path: str, emb_dim: int = 200, pool_output: bool = True):
        super().__init__()
        from CBraMod_main.models.cbramod import CBraMod
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seq_len = time_samples // patch_size
        print(f"CBraModBackbone: Using in_dim={in_dim}, time_samples={time_samples}, patch_size={patch_size}, seq_len={seq_len}")
        self.model = CBraMod(in_dim=in_dim, out_dim=emb_dim, d_model=emb_dim, seq_len=seq_len).to(self.device)
        
        weights = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # ! Inspect the weights! maybe we mess stho up with the initialization / classifier head: 
        state_dict = weights['model_state_dict'] if 'model_state_dict' in weights else weights
        print("Inspecting loaded weights for CBraModBackbone:")
        for name, tensor in state_dict.items():
            component = "BACKBONE" if 'backbone.' in name else "OTHER"
            print(f"[{component:<20}] {name:<60} Shape: {list(tensor.shape)}, trainable: {tensor.requires_grad}")
            
        if any('backbone.' in k for k in weights.keys()):
            # if weights were saved with a backbone prefix then strip it
            weights = {k.replace('backbone.', ''): v for k, v in weights.items()}
            
            # remove also the classifier. head if present
            weights = {k: v for k, v in weights.items() if not k.startswith('classifier.')}
            
        self.model.load_state_dict(weights, strict=False) #! maybe casino if pretrained weights have different classifier head
        # identity head + freeze default (trainer can unfreeze if requested)
        self.model.proj_out = nn.Identity()
        self.feature_dim = emb_dim
        self.pool_output = pool_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # normalize per-channel per-trial like your extractor

        #x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        # patchify exactly like your extractor
        T = x.shape[-1]
        # infer patch_size from how CBraMod was constructed
        seq_len = self.model.seq_len if hasattr(self.model, "seq_len") else max(1, T // 8)
        patch_size = T // seq_len
        # x shape : (B, C, T)
        patched = segment_to_patches(x, patch_size).to(self.device)  # (B, C, S, patch_size)
        out = self.model.forward(patched)            # (B, C, S, d_model)
        
        if self.pool_output:
            # Stage 2 (DVAE): Return 2D pooled vector (B, d_model)
            return out.mean(dim=(1, 2))
        else:
            # Stage 1 (External Classification): Return 4D feature map (B, C, S, d_model)
            return out

# ============================================================================
# Backbone Factory
# ============================================================================

def create_backbone(args, data_shape: Tuple[int, int]) -> Tuple[Optional[nn.Module], int, bool]:
    """
    Create appropriate backbone based on training mode.
    
    Returns:
        backbone: Feature extractor module (or None for Mode A)
        feature_dim: Output feature dimension
        requires_grad: Whether backbone should be trainable initially
    """
    mode = get_training_mode(args)
    n_chans, n_times = data_shape
    print(f"Creating backbone for mode: {mode}, data shape: {data_shape}, n_chans: {n_chans}, n_times: {n_times}")
    
    # MODE E: Signal Processing Features
    if mode == TrainingMode.SIGNAL_FEATURES:
        
        # sfreq ACHTUNG LIA 128HZ HARD CODED, should be passed from args or data info
        backbone = SignalProcessingBackbone(
            n_chans=n_chans, n_times=n_times, sfreq=128.0 
        )
        # Feature dimension determined on first forward pass. Use placeholder 200 initially.
        feature_dim = 200 # TODO make it comparable to cbramod/labram output dim!!!!!! 
        requires_grad = False # Fixed, non-learnable feature extraction
        logging.info(f"Mode E: Created fixed signal processing backbone (Input: {n_chans}x{n_times})")
        return backbone, feature_dim, requires_grad
    
    # MODE A: Learnable Projector
    if mode == TrainingMode.RAW_LEARNABLE:
        backbone = LearnableRawProjector(
            n_chans=n_chans,
            n_times=n_times,
            hidden_dim=args.projector_hidden_dim,
            target_dim=200,
            dropout=args.projector_dropout
        )
        feature_dim = 200
        requires_grad = True  # Always trainable in Mode A
        logging.info("Mode A: Created learnable raw projector")
    
    # MODES B/C/D: Pretrained Backbone
    else:
        # Determine patch size
        if args.dataset == 'sleepedfx':
            patch_size = 60
        elif args.dataset == 'MI_eeg':
            patch_size = 200 # get_optimal_patch_size(n_times)
        else:
            patch_size = 256
        
        # Create backbone
        if args.backbone == 'labram':
            backbone = LabramBackbone(
                n_chans=n_chans,
                n_times=n_times,
                weights_path=args.backbone_weights,
                patch_size=patch_size,
                emb_size=200
            )
        elif args.backbone == 'cbramod':
            in_dim = patch_size if args.dataset == 'MI_eeg' else 200
            backbone = CBraModBackbone(
                in_dim=in_dim,
                time_samples=n_times,
                patch_size=patch_size,
                weights_path=args.backbone_weights,
                emb_dim=200
            )
        else:
            raise ValueError(f"Unknown backbone: {args.backbone}")
        
        feature_dim = 200
        
        # Determine initial trainability
        if mode == TrainingMode.FROZEN_BACKBONE or mode == TrainingMode.CLASSIFIER_ONLY:
            requires_grad = False
            logging.info(f"Mode B: Created frozen {args.backbone} backbone")
        elif mode == TrainingMode.TWO_STAGE_DISJOINT:
            requires_grad = True  # Will be trained in Stage 1
            logging.info(f"Mode C: Created {args.backbone} backbone (will finetune in Stage 1)")
        elif mode == TrainingMode.JOINT_THEN_FROZEN:
            requires_grad = True  # Joint training initially
            logging.info(f"Mode D: Created {args.backbone} backbone (joint training initially)")
        
        # Apply initial freeze if needed
        if not requires_grad:
            backbone.eval()
            for param in backbone.parameters():
                param.requires_grad_(False)
                param.requires_grad = False
            # set model to eval mode
            print("Backbone frozen - set to eval mode")
            
    
    return backbone, feature_dim, requires_grad


class ExternalClassifierHead(nn.Module):
    def __init__(self, classifier_type: str, feature_dim: int, num_of_classes: int, dropout: float = 0.5):
        super().__init__()
        # Implementing the 'avgpooling_patch_reps' logic for simplicity as in cbramod git
        if classifier_type == 'avgpooling_patch_reps':
            self.head = nn.Sequential(
                Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(feature_dim, num_of_classes), # feature_dim is 200 for cbramod
            )
        else:
            raise NotImplementedError(f"Classifier type {classifier_type} not implemented for Stage 1.")

    def forward(self, x):
        return self.head(x)

class BackboneFinetuneTainer:
    """
    Stage 1 trainer for Mode C: Finetunes the backbone using either the 
    internal DVAE classification heads or a temporary external classifier.
    """
    def __init__(self, model: DisentangledEEGModel, loss_fn: DisentanglementLoss,
                 lr: float, device: str = 'cuda', 
                 use_external_classifier: bool = True, 
                 include_subject_classifiers: bool = True,
                 num_classes_subject: int = 10, num_classes_task: int = 4):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.device = device
        self.use_external_classifier = use_external_classifier
        self.include_subject_classifiers = include_subject_classifiers  
        
        
        # Initialize external classifiers if needed
        if self.include_subject_classifiers:
            self.temp_cls_subject = None
        self.temp_cls_task = None
        
        # 1. Freeze everything EXCEPT feature_extractor and classifier heads
        trainable_params: List[torch.nn.Parameter] = []
        
        for name, param in self.model.named_parameters():
            # Always freeze DVAE components
            if not name.startswith('feature_extractor'):
                param.requires_grad = False
            
            # Keep feature_extractor trainable (always trained in Stage 1)
            else:
                param.requires_grad = True
                # trainable_params.append(param)

            if self.include_subject_classifiers:
                self.temp_cls_subject = ExternalClassifierHead(
                classifier_type='avgpooling_patch_reps', feature_dim=200, num_of_classes=num_classes_subject).to(device)
            self.temp_cls_task = ExternalClassifierHead(
                classifier_type='avgpooling_patch_reps', feature_dim=200, num_of_classes=num_classes_task).to(device)
            
            # Add their parameters to the optimizer
            #trainable_params.extend(list(self.temp_cls_subject.parameters()))
            # trainable_params.extend(list(self.temp_cls_task.parameters()))
        

        trainable_params = list(p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)
        
        # --- 3. Logging ---    
        mode_log = "EXTERNAL" if self.use_external_classifier else "INTERNAL (DVAE)"
        # logging.info(f"Stage 1 Mode: {mode_log}. Finetuning {sum(p.numel() for p in trainable_params):,} parameters.")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters: {total_params:,}")
        total_total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters (trainable + frozen): {total_total_params:,}")
    
    def train_epoch(self, dataloader) -> Dict[str, float]:
        """Train one epoch on classification only"""
        # print shape of input data
        for batch in dataloader:
            print(f"Input batch shape: {batch[1].shape}")
            break
     
        
        self.model.train()
        
        if self.use_external_classifier:
            if self.include_subject_classifiers:
                self.temp_cls_subject.train()
            self.temp_cls_task.train()
            
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(dataloader):
            num_batches += 1
            features = batch[1].to(self.device)
            labels = {
                'subject': batch[2].to(self.device),
                'task': batch[3].to(self.device)
            }
            
            # Forward pass (only extract + classify)
            if self.model.feature_extractor is not None:
                extracted_feats = self.model.extract_features(features)
            else:
                extracted_feats = features
            
            if self.use_external_classifier:
                # Use TEMPORARY external heads
                if self.include_subject_classifiers:
                    sub_logits = self.temp_cls_subject(extracted_feats)
                task_logits = self.temp_cls_task(extracted_feats)
                
                # Combine logits into the expected structure for loss calculation
                if self.include_subject_classifiers:
                    logits_dict = {
                        'subject': {'logits': sub_logits},
                        'task': {'logits': task_logits}
                    }
                else:
                    logits_dict = {
                        'task': {'logits': task_logits}
                    }
                
            else:
                # Use ORIGINAL DIVA internal heads (gradients flow through them if trainable)
                # The model's 'encode' method handles the VAE/Classifier split internally (VAE stuff is not trainable, so just classification)
                logits_dict = self.model.encode(extracted_feats)
            
            # Compute only classification losses
            cls_loss = 0
            for name, enc_output in logits_dict.items():
                if 'logits' in enc_output and name in labels:
                    loss = F.cross_entropy(enc_output['logits'], labels[name])
                    epoch_losses[f'cls_{name}'] += loss.item()
                    cls_loss += loss
            #print("SO DANGEROUS; NOT ENTER HERE")
            # Backward
            self.optimizer.zero_grad(set_to_none=True)
            cls_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            epoch_losses['total'] += cls_loss.item()
        
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
    def train(self, train_loader, num_epochs: int, wandb_run):
        """Run Stage 1 training"""
        logging.info(f"\n{'='*60}")
        logging.info(f"STAGE 1: Backbone Finetuning ({num_epochs} epochs)")
        logging.info(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            metrics = self.train_epoch(train_loader)
            logging.info(f"Stage 1 Epoch {epoch+1}/{num_epochs} | "
                        f"Loss: {metrics['total']:.4f}")
            
        # Add logging to wandb
        if wandb_run is not None:
            wandb_run.log({f"stage1/{k}": v for k, v in metrics.items()}, step=epoch)
        
        logging.info("\nStage 1 Complete - Freezing Backbone\n")
        
        # Freeze feature extractor for Stage 2
        if self.model.feature_extractor is not None:
            for param in self.model.feature_extractor.parameters():
                param.requires_grad = False
                
        