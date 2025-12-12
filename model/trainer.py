import torch 
import numpy as np
import wandb
from utils import *
from model.backbones import *
from model.architecture import *
from pathlib import Path
from tqdm import tqdm

class DVAETrainer:
    """
    Main DVAE trainer - handles Modes A, B, C (Stage 2), D
    """
    def __init__(self, model: DisentangledEEGModel, loss_fn: DisentanglementLoss,
                 optimizer: torch.optim.Optimizer, device: str = 'cuda',
                 save_dir: str = 'experiments', mode: TrainingMode = None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = Path(save_dir)
        self.mode = mode
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_history = []
        self.val_history = []
    

    def tmp_split_optimizer(self, train_backbone: bool = False):
        current_lr = self.optimizer.param_groups[0]['lr']

        # use 2 learning rates for backbone and for the rest
        backbone_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'feature_extractor' in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        
        # self.optimizer = torch.optim.AdamW(trainable_params, lr=current_lr, weight_decay=0.01)
        if train_backbone:
            self.optimizer = torch.optim.AdamW([
                {'params': backbone_params, 'lr': current_lr},
                {'params': other_params, 'lr': current_lr * 5}
            ], weight_decay=5e-2)
        else:
            self.optimizer = torch.optim.AdamW([
                {'params': other_params, 'lr': current_lr}
            ], weight_decay=5e-2)
        print(f"Current learning rate: {current_lr}")

    def freeze_backbone(self):
        """Freeze backbone (called in Mode D at specified epoch)"""
        if self.model.feature_extractor is None:
            return
        
        logging.info(f"\n{'='*60}")
        logging.info("FREEZING BACKBONE (Mode D Transition)")
        logging.info(f"{'='*60}\n")
        
        for param in self.model.feature_extractor.parameters():
            param.requires_grad = False
        
        self.model.config.freeze_feature_extractor = True
        
        # Reinitialize optimizer without backbone params
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        
        logging.info(f"Trainable parameters reduced to {sum(p.numel() for p in trainable_params):,}\n")
    
    
    @torch.no_grad()
    def validate_classifier_only(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model in Classifier Only mode.
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        all_predictions = []
        all_labels = []

        for batch in dataloader:
            features = batch[1].to(self.device)
            labels = batch[3].to(self.device)  # Task labels

            outputs = self.model(features) # outputs is directly the logits here
            all_predictions.append(outputs.argmax(dim=-1).cpu())
            all_labels.append(labels.cpu())
            losses = self.loss_fn.compute_loss_classification_only(outputs, {'task': labels}, self.model)

            # Accumulate losses
            for k, v in losses.items():
                if torch.is_tensor(v):
                    epoch_losses[k].append(v.item())

        # --- Average Losses ---
        metrics = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # --- Compute Classification Accuracy ---
        if all_predictions:
            preds = torch.cat(all_predictions)
            labs = torch.cat(all_labels)
            accuracy = (preds == labs).float().mean().item()
            metrics[f'accuracy_task'] = accuracy
                
        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model, including Subject Embedding Consistency (Metric Learning) 
        for generalization quality and Task Classification Accuracy.
        """
        self.model.eval()
        epoch_losses = defaultdict(list)
        
        # 1. Collect all data for Metric Learning and Loss Accumulation
        all_predictions = {name: [] for name in self.model.encoder_names 
                        if self.model.config.encoders[name].num_classes is not None}
        all_labels = {name: [] for name in all_predictions.keys()}
        
        all_z_subject = []
        all_subject_labels = []

        for batch in dataloader:
            # Data from CustomLoader/CustomLoaderSleep/CustomLoaderMI is (indices, data, subjects, tasks, runs)
            features = batch[1].to(self.device)
            labels = {
                'subject': batch[2].to(self.device),
                'task': batch[3].to(self.device)
            }

            outputs = self.model(features)
            # print outputs and labels in readable format for debugging
            #print("Outputs:", {k: v.keys() for k, v in outputs['encoded'].items()})
            #print("Labels:", labels)
            losses = self.loss_fn.compute_loss(outputs, labels, self.model)

            # Accumulate losses
            for k, v in losses.items():
                # Ensure V is a tensor before calling .item() and appending
                if torch.is_tensor(v):
                    epoch_losses[k].append(v.item())
            
            # Collect embeddings and labels for metric calculation
            if 'subject' in outputs['encoded']:
                z_subject = outputs['encoded']['subject']['z']
                all_z_subject.append(z_subject.cpu())
                all_subject_labels.append(labels['subject'].cpu())
                
            # Collect predictions for classification metrics (Task)
            for name in all_predictions.keys():
                if 'logits' in outputs['encoded'][name]:
                    preds = outputs['encoded'][name]['logits'].argmax(dim=-1)
                    all_predictions[name].append(preds.cpu())
                    all_labels[name].append(labels[name].cpu())

        # --- 2. Average Losses ---
        metrics = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        
        # --- 3. Compute Subject Embedding Consistency (Metric Learning) ---
        if all_z_subject:
            Z = torch.cat(all_z_subject)  # All subject embeddings (N_samples, Latent_dim)
            Y = torch.cat(all_subject_labels) # All subject IDs (N_samples)
            
            # Ensure data is on the correct device and is float
            device = self.device
            Z = Z.float().to(device)
            Y = Y.to(device) 

            # We must have at least two samples from two different subjects for the metric to be meaningful.
            if len(Y.unique()) > 1 and Z.shape[0] > 1:
                
                # Compute full pairwise cosine similarity matrix (N x N)
                Z_norm = F.normalize(Z, p=2, dim=1)
                similarity_matrix = torch.matmul(Z_norm, Z_norm.transpose(0, 1))

                # 1. Create Identity Mask: M[i, j] = True if Y[i] == Y[j] (Same Subject)
                identity_matrix = (Y[:, None] == Y[None, :]) 
                
                # 2. Exclude Self-Similarity: Diagonal elements (i=j) must be ignored
                mask_upper_triangular = torch.triu(torch.ones_like(similarity_matrix, dtype=torch.bool), diagonal=1)

                # --- Intra-Subject Similarity ---
                # Select similarities where (Subject IDs match AND it's in the upper triangle to avoid duplicates)
                intra_mask = identity_matrix * mask_upper_triangular
                intra_subject_similarities = similarity_matrix[intra_mask]

                # --- Inter-Subject Similarity ---
                # Select similarities where (Subject IDs DO NOT match AND it's in the upper triangle)
                inter_mask = (~identity_matrix) * mask_upper_triangular
                inter_subject_similarities = similarity_matrix[inter_mask]

                # --- Calculate Metrics ---
                
                avg_intra_sim = intra_subject_similarities.mean().item() if intra_subject_similarities.numel() > 0 else 0.0
                avg_inter_sim = inter_subject_similarities.mean().item() if inter_subject_similarities.numel() > 0 else 0.0
                
                metrics['metric_avg_intra_subject_similarity'] = avg_intra_sim
                metrics['metric_avg_inter_subject_similarity'] = avg_inter_sim
                
                # The key metric: should be positive for good subject representation generalization
                consistency_score = avg_intra_sim - avg_inter_sim
                metrics['metric_subject_consistency_score'] = consistency_score
            
        # --- 4. Compute Classification Accuracies (for Task only) ---
        for name in all_predictions.keys():
            if name == 'task' and all_predictions[name]:
                preds = torch.cat(all_predictions[name])
                labs = torch.cat(all_labels[name])
                accuracy = (preds == labs).float().mean().item()
                metrics[f'accuracy_{name}'] = accuracy
                
        # Remove subject accuracy metric if present, as it is meaningless for unseen subjects
        if 'accuracy_subject' in metrics:
            metrics.pop('accuracy_subject')
                
        return metrics
    
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch in tqdm(dataloader):
            num_batches += 1
            features = batch[1].to(self.device)
            labels = {
                'subject': batch[2].to(self.device),
                'task': batch[3].to(self.device)
            }
            
            # --- 1. Forward Pass (Reconstruction/Encoding) ---
            # print(f"Training Batch {num_batches}: features shape {features.shape}")
            outputs = self.model(features)
            
            
            # --- 2. Cross-Generation for Loss (if enabled) ---
            # These outputs are now needed for compute_loss, but must be computed first
            
            # TODO THIS IS MESSED UP LIA; FIX IT! 
            if self.loss_fn.config.cross_subject_intra_class or self.loss_fn.config.cross_subject_cross_class:
                # Generate a second batch (B) by cyclically shifting the first batch (A) for pairing.
                # This ensures A and B are different samples in the batch.
                features_B = torch.roll(features, shifts=1, dims=0)
                print("Features B shape:", features_B.shape)
                labels_B = {k: torch.roll(v, shifts=1, dims=0) for k, v in labels.items()}
                print("Labels B subject shape:", labels_B['subject'].shape)
                encoded_B = self.model.encode(features_B.detach()) # warum mag er die shape hier nicht?? 
                print("Encoded B keys:", encoded_B.keys())
                print("Encoded B subject z shape:", encoded_B['subject']['z'].shape)

                if self.loss_fn.config.cross_subject_intra_class:
                    # Intra-Class: Swap subject (S_B) but keep task/noise (T_A, N_A). Target is Feature B.
                    outputs['cross_intra_reconstruction'] = self.model.generate_swapped_reconstruction(
                        encoded_A=outputs['encoded'], encoded_B=encoded_B,
                        swap_source_name='subject', swap_target_name='subject'
                    )
                    outputs['cross_intra_target'] = features_B
                    
                if self.loss_fn.config.cross_subject_cross_class:
                    # Cross-Class: Swap subject (S_B) AND task (T_B). Target is Feature B.
                    # This is complex. For simplicity, let's keep it to the simplest form of generating 
                    # a sample that should be far from the original. We will swap subject and keep task from B.
                    # This attempts to generate A-like feature but with B's subject/task.
                    outputs['cross_cross_reconstruction'] = self.model.generate_swapped_reconstruction(
                        encoded_A=outputs['encoded'], encoded_B=encoded_B,
                        swap_source_name='subject', swap_target_name='subject'
                    ) # Note: For true cross-class, we'd need a separate task swap as well.
                    outputs['cross_cross_target'] = features_B
            
            # --- 3. Compute ALL Losses (including WGAN components) ---
            if self.mode == TrainingMode.CLASSIFIER_ONLY:
                losses = self.loss_fn.compute_loss_classification_only(outputs, labels, self.model)
            else:
                losses = self.loss_fn.compute_loss(outputs, labels, self.model)
            # --- 4. Train Discriminator (D) ---
            if self.loss_fn.config.adversarial:
                # D loss = Adv_D loss + GP
                loss_D = losses['adversarial_discriminator'] + \
                         self.loss_fn.config.lambda_gp * losses['gradient_penalty']
                
                self.optimizer_D.zero_grad(set_to_none=True)
                # We need to retain the graph for the reconstruction/features used in the Generator update
                #loss_D.backward(retain_graph=True) 
                #self.optimizer_D.step()
                
            # --- 5. Train Generator/Encoders (G) --- 
            self.optimizer.zero_grad(set_to_none=True)
            
            # G loss = Total VAE/CLS/Consistency Loss + Adv_G loss
            # Crucially, we must exclude D-specific losses (Adv_D, GP)
            loss_G = losses['total']
            if self.loss_fn.config.adversarial:
                # Subtract D's loss terms that were incorrectly added to the total.
                # total_loss in compute_loss is: L_VAE + L_CLS + L_CONSISTENCY + L_Adv_G + L_Adv_D + L_GP
                # We want: L_VAE + L_CLS + L_CONSISTENCY + L_Adv_G
                loss_G -= losses['adversarial_discriminator']
                loss_G -= self.loss_fn.config.lambda_gp * losses['gradient_penalty']
                
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=100.0)
            self.optimizer.step()
            
            # Accumulate losses (use loss_G for logging total loss for the model)
            for k, v in losses.items():
                # We only log the D loss for logging, not for adding to the G total
                if self.loss_fn.config.adversarial and k in ('adversarial_discriminator', 'gradient_penalty'):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += v.item()
            
            torch.cuda.empty_cache()  # Clear cache to avoid OOM



        # Average losses
        return {k: v / num_batches for k, v in epoch_losses.items()}
    
 
    
    def train(self, train_loader, val_loader, num_epochs: int, 
              freeze_at_epoch: Optional[int] = None, wandb_run=None):
        """
        Main training loop
        
        Args:
            freeze_at_epoch: For Mode D, epoch to freeze backbone
        """
        best_val_loss = float('inf')
        
        stage_name = {
            TrainingMode.RAW_LEARNABLE: "Raw Learnable Projector + DVAE",
            TrainingMode.FROZEN_BACKBONE: "DVAE (Frozen Backbone)",
            TrainingMode.TWO_STAGE_DISJOINT: "STAGE 2: DVAE (Frozen Backbone)",
            TrainingMode.JOINT_THEN_FROZEN: "Joint Training → Frozen",
            TrainingMode.SIGNAL_FEATURES: "DVAE on Pre-extracted Signal Features",
            TrainingMode.CLASSIFIER_ONLY: "Classifier Only on Frozen Backbone Features"
        }.get(self.mode, "DVAE Training")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"{stage_name}")
        logging.info(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Mode D: Freeze backbone at specified epoch
            if (self.mode == TrainingMode.JOINT_THEN_FROZEN and 
                freeze_at_epoch is not None and 
                epoch == freeze_at_epoch):
                self.freeze_backbone()
            
            logging.info(f"\nEpoch {epoch+1}/{num_epochs}")
            
            train_metrics = self.train_epoch(train_loader)
            self.train_history.append(train_metrics)
            
            if self.mode == TrainingMode.CLASSIFIER_ONLY:
                val_metrics = self.validate_classifier_only(val_loader)
            else:
                val_metrics = self.validate(val_loader)

            self.val_history.append(val_metrics)
            
            if wandb_run is not None:
                # Prepara il dizionario di logging unendo Training e Validation
                log_data = {
                    'epoch': epoch + 1,
                    # Aggiungi prefissi per distinguere le metriche
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                }
                wandb.log(log_data)
            
            logging.info(f"Train Loss: {train_metrics['total']:.4f}")
            logging.info(f"Val Loss: {val_metrics['total']:.4f}")
            
            for key in val_metrics:
                if 'accuracy' in key:
                    logging.info(f"Val {key}: {val_metrics[key]:.4f}")
            
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                self.save_checkpoint('best_model.pt', epoch, val_metrics)
            
            # save the last model every epoch
            self.save_checkpoint('last_model.pt', epoch, val_metrics)
            self.save_model_weights('last_model_weights.pt')
                
            # Log val history (val_metrics) every epoch
            logging.info(f"Validation history: {self.val_history}")

    
    def save_checkpoint(self, filename: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'config': self.model.config
        }
        torch.save(checkpoint, self.save_dir / filename)
        
    # a save checkpoint which just contains the weights of the model (no optimizer, no history)
    def save_model_weights(self, filename: str):
        """Save only model weights."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
        self.val_history = checkpoint['val_history']
        return checkpoint['epoch'], checkpoint['metrics']
