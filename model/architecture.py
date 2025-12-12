from utils.helper_functions import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from torch.autograd import grad
import scipy
import scipy.signal as signal
from scipy.stats import entropy
from scipy.fft import fft, fftfreq
from einops.layers.torch import Rearrange



# ============================================================================
# Configuration System
# ============================================================================
# (ModelConfig, LossConfig, EncoderConfig classes follow here)

@dataclass
class EncoderConfig:
    """Configuration for a single encoder."""
    name: str
    enabled: bool = True
    latent_dim: int = 32
    num_classes: Optional[int] = None  # None = no classifier
    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])



@dataclass
class LossConfig:
    """Configuration for all losses."""
    # Reconstruction losses
    self_reconstruction: bool = True
    self_reconstruction_weight: float = 1.0
    
    # VAE losses
    kl_divergence: bool = True
    kl_weight: float = 0.05  # β parameter
    noise_kl_weight: float = 1.0  # Separate weight for noise encoder
    
    # Classification losses
    classification: bool = True
    classification_weight: float = 1.0
    
    # Generation losses
    self_generation: bool = False
    self_generation_weight: float = 0.5
    
    # Consistency losses
    latent_consistency: bool = False
    latent_consistency_weight: float = 0.5
    self_cycle: bool = False
    self_cycle_weight: float = 0.5
    
    # Cross-subject losses
    cross_subject_intra_class: bool = False
    cross_subject_intra_class_weight: float = 0.3
    cross_subject_cross_class: bool = False
    cross_subject_cross_class_weight: float = 0.3
    
    # Cross-class cycle consistency
    cross_class_cycle_feature: bool = False
    cross_class_cycle_feature_weight: float = 0.3
    cross_class_cycle_signal: bool = False
    cross_class_cycle_signal_weight: float = 0.3
    
    # Knowledge distillation
    knowledge_distillation: bool = False
    kd_weight: float = 0.5
    kd_temperature: float = 3.0
    
    # Adversarial losses (WGAN-GP)
    adversarial: bool = False
    adversarial_weight: float = 1.0
    lambda_gp: float = 10.0
    
    use_stft_loss: bool = False
    adaptive_balancing: bool = False






@dataclass
class ModelConfig:
    """Complete model configuration."""
    # Feature extractor settings
    n_channels: int = 64
    time_samples: int = 2048
    freeze_feature_extractor: bool = False
    
    # Encoders (can add/remove as needed)
    encoders: Dict[str, EncoderConfig] = field(default_factory=lambda: {
        'subject': EncoderConfig('subject', enabled=True, latent_dim=32, num_classes=10),
        'task': EncoderConfig('task', enabled=True, latent_dim=64, num_classes=4),
        'noise': EncoderConfig('noise', enabled=True, latent_dim=16, num_classes=None),
        'device': EncoderConfig('device', enabled=False, latent_dim=16, num_classes=3),
        'session': EncoderConfig('session', enabled=False, latent_dim=16, num_classes=5),
    })
    
    # Loss configuration
    loss_config: LossConfig = field(default_factory=LossConfig)
    
    # Generator settings (default_factory: ensures each instance gets its own fresh copy of the list)
    generator_hidden_dims: List[int] = field(default_factory=lambda: [256, 512])




# ============================================================================
# LearnableRawProjector (Mode A)
# ============================================================================

class LearnableRawProjector(nn.Module):
    """
    Learnable projector for raw EEG data (Mode A).
    Includes normalization and better architecture.
    """
    def __init__(self, n_chans: int, n_times: int, hidden_dim: int = 512,
                 target_dim: int = 200, dropout: float = 0.1):
        super().__init__()
        
        input_dim = n_chans * n_times
        self.feature_dim = target_dim
        
        # Learnable normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Deeper projector with residual connection
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, target_dim)
        )
        
        # Initialize with smaller weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) raw EEG
        Returns:
            features: (B, target_dim)
        """
        B, C, T = x.shape
        
        # Flatten and normalize
        x_flat = x.reshape(B, -1)  # (B, C*T)
        x_norm = self.input_norm(x_flat)
        
        # Project
        features = self.projector(x_norm)
        return features
    

# ============================================================================
# SignalProcessingBackbone (Mode B)
# ============================================================================    

class SignalProcessingBackbone(nn.Module):
    """
    Extracts traditional time-domain and frequency-domain features 
    and returns a fixed-length feature vector.
    """
    def __init__(self, n_chans: int, n_times: int, sfreq: float = 128.0):
        super().__init__()
        
        self.sfreq = sfreq
        self.n_chans = n_chans
        self.feature_names = None  # To be populated after first forward pass
        
        # The feature dimension is determined by the number of features extracted per channel.
        # Based on the advanced extraction logic, this is approximately 10-15 features/channel.
        # We will dynamically set feature_dim after the first pass.
        self.feature_dim = 0 
        
    def _extract_channel_features(self, ch_data: np.ndarray) -> Dict[str, float]:
        """Core logic to extract advanced temporal and spectral features for one channel."""
        features = {}
        
        # 1. TEMPORAL FEATURES
        features['mean'] = np.mean(ch_data)
        features['std'] = np.std(ch_data)
        features['skew'] = scipy.stats.skew(ch_data)
        features['kurtosis'] = scipy.stats.kurtosis(ch_data)
        features['rms'] = np.sqrt(np.mean(ch_data**2))
        
        # 2. FREQUENCY FEATURES (PSD via Welch's method)
        freq_bands = {
            'delta': (0.5, 4), 'theta': (4, 8), 'alpha': (8, 12),
            'beta': (12, 30), 'gamma': (30, 45)
        }
        
        # nperseg should be <= length of data. 
        nperseg = min(256, len(ch_data)) # TODO make this data dependent?
        freqs, psd = signal.welch(ch_data, fs=self.sfreq, nperseg=nperseg)
        total_power = np.trapezoid(psd, freqs)

        for band_name, (low_freq, high_freq) in freq_bands.items():
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            band_power = np.trapezoid(psd[band_mask], freqs[band_mask])
            features[f'{band_name}_power'] = band_power
            
            # Relative Power
            features[f'{band_name}_relative'] = band_power / total_power if total_power > 1e-6 else 0
            
        # 3. ENTROPY & PEAK
        features['entropy'] = entropy(np.histogram(ch_data, bins=10)[0])
        peak_freq_idx = np.argmax(psd)
        features['peak_freq'] = freqs[peak_freq_idx]

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts (B, C, T) raw EEG segments into (B, F) feature vectors.
        """
        B, C, T = x.shape
        x_np = x.cpu().numpy()
        all_features_np = []
        
        if C != self.n_chans:
             logging.warning(f"Channel count mismatch: {C} in data vs {self.n_chans} in config.")

        for seg_idx in range(B):
            segment_features = {}
            # Iterate through channels and extract features
            for ch_idx in range(C):
                try:
                    ch_features = self._extract_channel_features(x_np[seg_idx, ch_idx, :])
                    for k, v in ch_features.items():
                        segment_features[f'ch{ch_idx}_{k}'] = v
                except Exception as e:
                    logging.error(f"Error extracting features for segment {seg_idx}, channel {ch_idx}: {e}")
                    # Insert default zeros if calculation fails
                    if not segment_features: segment_features = {f'ch{ch_idx}_zero': 0.0}
            
            all_features_np.append(segment_features)

        # 4. Final Conversion to Tensor and Dimension Check
        if not all_features_np:
            return torch.zeros(B, self.feature_dim)
            
        # Convert list of dicts to NumPy array (ensuring consistent feature ordering)
        if self.feature_names is None:
            self.feature_names = sorted(all_features_np[0].keys())
            self.feature_dim = len(self.feature_names)
        
        feature_vector = np.array([
            [d.get(name, 0.0) for name in self.feature_names] for d in all_features_np
        ], dtype=np.float32)
        
        return torch.from_numpy(feature_vector).to(x.device)



# ============================================================================
# VAE Encoder / Generator / Discriminator / DisentangledEEGModel 
# ============================================================================
class VAEEncoder(nn.Module):
    """VAE encoder with optional classifier head."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int], 
                 num_classes: Optional[int] = None):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Encoder backbone
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # VAE heads
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Optional classifier head
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(latent_dim, num_classes)
            )
        else:
            self.classifier = None
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Reparameterization
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        result = {'z': z, 'mu': mu, 'logvar': logvar}
        
        # Classification
        if self.classifier is not None:
            logits = self.classifier(z)
            result['logits'] = logits
        
        return result


# ============================================================================
# Generator
# ============================================================================

class Generator(nn.Module):
    """Generator to reconstruct features from concatenated latent codes."""
    
    def __init__(self, latent_dims: Dict[str, int], output_dim: int, 
                 hidden_dims: List[int]):
        super().__init__()
        
        total_latent_dim = sum(latent_dims.values())
        
        layers = []
        prev_dim = total_latent_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, latent_codes: Dict[str, torch.Tensor]) -> torch.Tensor:
        z_list = [latent_codes[name] for name in sorted(latent_codes.keys())]
        z_concat = torch.cat(z_list, dim=-1)
        return self.decoder(z_concat)

# ============================================================================
# Discriminator
# ============================================================================
class Discriminator(nn.Module):
    """Simple discriminator for adversarial training."""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.LayerNorm(h),
                nn.LeakyReLU(0.2)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# ============================================================================
# Complete Disentangled Model
# ============================================================================

class DisentangledEEGModel(nn.Module):
    """Modular disentangled representation learning model."""
    
    def __init__(self, feature_extractor: Optional[nn.Module], feature_dim: int, 
                 config: ModelConfig):
        super().__init__()
        
        self.config = config
        self.feature_extractor = feature_extractor
        self.discriminator = Discriminator(feature_dim)
        
        if config.freeze_feature_extractor and self.feature_extractor is not None:
           for param in self.feature_extractor.parameters():
               param.requires_grad = False
        
        # Create enabled encoders
        self.encoders = nn.ModuleDict()
        self.encoder_names = []
        
        for name, enc_config in config.encoders.items():
            if enc_config.enabled:
                self.encoders[name] = VAEEncoder(
                    feature_dim,
                    enc_config.latent_dim,
                    enc_config.hidden_dims,
                    enc_config.num_classes
                )
                self.encoder_names.append(name)
        
        # Generator
        latent_dims = {name: config.encoders[name].latent_dim 
                      for name in self.encoder_names}
        self.generator = Generator(latent_dims, feature_dim, 
                                   config.generator_hidden_dims)
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is None:
            return x
        if self.config.freeze_feature_extractor:
            with torch.no_grad():
                x_new = x.detach().clone()
                features = self.feature_extractor(x_new)
                
                return features
        return self.feature_extractor(x)
    
    def set_feature_pooling(self, enable_pooling: bool):
        """Allows external control over the backbone's output dimensionality."""
        if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'pool_output'):
            self.feature_extractor.pool_output = enable_pooling
            logging.info(f"Feature Extractor output pooling set to: {enable_pooling}")
        else:
            logging.warning("Feature extractor or pool_output attribute not found.")
            
    def set_required_grad_for_dvae(self, requires_grad: bool = True):
        """Enable/disable gradients for DVAE components (encoders + generator)."""
        for encoder in self.encoders.values():
            for param in encoder.parameters():
                param.requires_grad = requires_grad
        for param in self.generator.parameters():
            param.requires_grad = requires_grad
        for param in self.discriminator.parameters():
            param.requires_grad = requires_grad
    
    def generate_swapped_reconstruction(self, 
                                        encoded_A: Dict[str, Dict[str, torch.Tensor]], 
                                        encoded_B: Dict[str, Dict[str, torch.Tensor]],
                                        swap_source_name: str,
                                        swap_target_name: str) -> torch.Tensor:
        """
        Generates a feature vector by taking latent codes from A, 
        but substituting 'swap_target_name' code from B.
        e.g., A = (z_subject_A, z_task_A, z_noise_A)
        B = (z_subject_B, z_task_B, z_noise_B)
        Swap Subject: -> (z_subject_B, z_task_A, z_noise_A)
        """
        latent_codes_swapped = {}
        # Iterate over all enabled latent codes (z)
        for name in self.encoder_names:
            if name == swap_source_name:
                # Target latent code comes from B (the thing we want to swap in)
                latent_codes_swapped[name] = encoded_B[name]['z']
            else:
                # Other latent codes come from A (the things we want to keep)
                latent_codes_swapped[name] = encoded_A[name]['z']
                
        return self.generator(latent_codes_swapped)
    
    def encode(self, features: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Encode features with all enabled encoders."""
        encoded = {}
        for name in self.encoder_names:
            encoded[name] = self.encoders[name](features)
        return encoded
    
    def decode(self, encoded: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Reconstruct from latent codes."""
        latent_codes = {name: enc['z'] for name, enc in encoded.items()}
        return self.generator(latent_codes)
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Complete forward pass."""
        # Extract features if a model is provided
        if self.feature_extractor is not None:
            features = self.extract_features(x)
        else:
            # Otherwise, assume x is already the features
            features = x
        
        # Encode
        features = features.detach().clone()
        encoded = self.encode(features)
        
        # Reconstruct
        reconstruction = self.decode(encoded)
        
        return {
            'features': features,
            'encoded': encoded,
            'reconstruction': reconstruction
        }



class SimpleFeaturesClassifier(nn.Module):
    """Modular disentangled representation learning model."""
    
    def __init__(self, feature_extractor: Optional[nn.Module], feature_dim: int, 
                 config: ModelConfig, num_classes: int = 4, train_backbone: bool = True):
        from .backbones import ExternalClassifierHead

        super().__init__()
        
        self.config = config
        self.feature_extractor = feature_extractor
        self.train_backbone = train_backbone

        self.set_feature_pooling(False)
        self.classifier = ExternalClassifierHead(classifier_type="avgpooling_patch_reps", feature_dim=feature_dim, num_of_classes=num_classes)
        
        # if config.freeze_feature_extractor and self.feature_extractor is not None:
        #    for param in self.feature_extractor.parameters():
        #        param.requires_grad = False
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is None:
            return x
        if self.config.freeze_feature_extractor:
            with torch.no_grad():
                
                return self.feature_extractor(x)
        return self.feature_extractor(x)
    
    def set_feature_pooling(self, enable_pooling: bool):
        """Allows external control over the backbone's output dimensionality."""
        if hasattr(self, 'feature_extractor') and hasattr(self.feature_extractor, 'pool_output'):
            self.feature_extractor.pool_output = enable_pooling
            logging.info(f"Feature Extractor output pooling set to: {enable_pooling}")
        else:
            logging.warning("Feature extractor or pool_output attribute not found.")
            
    def set_required_grad_for_classifier(self, requires_grad: bool = True):
        """Enable/disable gradients for Classifier."""
        for param in self.classifier.parameters():
            print(f"Setting requires_grad={requires_grad} for classifier param")
            param.requires_grad = requires_grad
    
    def set_required_grad_for_backbone(self, requires_grad: bool = True):
        """Enable/disable gradients for Backbone."""
        self.train_backbone = requires_grad
        for param in self.feature_extractor.parameters():
            print(f"Setting requires_grad={requires_grad} for backbone param")
            param.requires_grad = requires_grad
    
    
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        """Complete forward pass."""
        # Extract features if a model is provided
        if self.feature_extractor is not None:
            features = self.extract_features(x)
        else:
            # Otherwise, assume x is already the features
            features = x
        
        # Encode
        if self.train_backbone:
            features_new = features
        else:
            features_new = features.detach().clone()
        
        # Reconstruct
        logits = self.classifier(features_new)
        
        return logits

# create class to continue training a SimpleFeaturesClassifier
class SimpleFeaturesClassifierContinued(SimpleFeaturesClassifier):
    def __init__(self, feature_extractor: Optional[nn.Module], feature_dim: int, 
                 config: ModelConfig, num_classes: int = 4, previous_model_path: str = ''):
        super().__init__(feature_extractor, feature_dim, config, num_classes)
        
        # load previous model weights
        weights = torch.load(previous_model_path, map_location='cpu')
        if 'model_state_dict' in weights:
            weights = weights['model_state_dict']
        
        # load only the classifier weights
        classifier_weights = {k.replace('classifier.', ''): v for k, v in weights.items() if k.startswith('classifier.')}
        self.classifier.load_state_dict(classifier_weights, strict=False)
        print(f"Loaded classifier weights from {previous_model_path}")



class MLPClassifier(nn.Module):
    """Simple MLP classifier for features."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        layers.extend([Rearrange('b c s d -> b d c s'),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()])
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

# ============================================================================
# Loss Functions
# ============================================================================

class DisentanglementLoss(nn.Module):
    """Comprehensive loss function with all disentanglement objectives."""
    
    def __init__(self, config: LossConfig):
        super().__init__()
        self.config = config
        # Initialize adaptive state dictionary
        self.state = {'prev_losses': {}}
    
    def kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """KL divergence between N(mu, var) and N(0, 1)."""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    def self_reconstruction_loss(self, reconstruction: torch.Tensor, 
                                target: torch.Tensor) -> torch.Tensor:
        """MSE reconstruction loss.
        Maximizes the likelihood that the generated sample (x^hat_i) resembles the original input (x_i). 
        It is calculated in the time-frequency domain using the Short-Term Fourier Transform (STFT)"""
        return F.mse_loss(reconstruction, target)
    
    def classification_loss(self, logits: torch.Tensor, 
                          labels: torch.Tensor) -> torch.Tensor:
        """Cross-entropy classification loss."""
        return F.cross_entropy(logits, labels)
    
    def latent_consistency_loss(self, encoded1: Dict, encoded2: Dict) -> torch.Tensor:
        """Consistency between latent codes from same sample."""
        loss = 0
        count = 0
        for name in encoded1.keys():
            if name in encoded2:
                loss += F.mse_loss(encoded1[name]['mu'], encoded2[name]['mu'])
                count += 1
        return loss / count if count > 0 else torch.tensor(0.0)
    
    def self_cycle_loss(self, original: torch.Tensor, cycled: torch.Tensor) -> torch.Tensor:
        """Cycle consistency: encode -> decode -> encode should be consistent."""
        return F.mse_loss(original, cycled)
    
    def cross_subject_intra_class_loss(self, reconstruction: torch.Tensor,
                                      target_features: torch.Tensor) -> torch.Tensor:
        """Generate samples with different subjects but same task."""
        return F.mse_loss(reconstruction, target_features)
    
    def cross_subject_cross_class_loss(self, reconstruction: torch.Tensor,
                                      target_features: torch.Tensor) -> torch.Tensor:
        """Generate samples with different subjects and tasks."""
        # Should be different, so we maximize distance (minimize negative MSE)
        return -F.mse_loss(reconstruction, target_features)
    
    def knowledge_distillation_loss(self, student_logits: torch.Tensor,
                                   teacher_logits: torch.Tensor,
                                   temperature: float) -> torch.Tensor:
        """KD loss for regularization."""
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    def compute_loss(self, outputs: Dict[str, Any], labels: Dict[str, torch.Tensor],
                    model: DisentangledEEGModel) -> Dict[str, torch.Tensor]:
        """Compute all enabled losses."""
        losses = {}
        total_loss = torch.tensor(0.0, device='cuda:0')
        

        encoded = outputs['encoded']
        reconstruction = outputs['reconstruction']
        features = outputs['features']
        
        # 1. Self-reconstruction loss
        if self.config.self_reconstruction:
            #if self.config.use_stft_loss:
            #    recon_loss = self.stft_mse(reconstruction, features)
            #else:
            #    recon_loss = F.mse_loss(reconstruction, features)
            recon_loss = F.mse_loss(reconstruction, features)
            losses['self_reconstruction'] = recon_loss
            total_loss += self.config.self_reconstruction_weight * recon_loss
        
        # 2. KL divergence for each encoder
        if self.config.kl_divergence:
            kl_total = 0
            for name, enc_output in encoded.items():
                kl_loss = self.kl_divergence(enc_output['mu'], enc_output['logvar'])
                losses[f'kl_{name}'] = kl_loss
                
                # Apply different weight for noise encoder
                weight = (self.config.noise_kl_weight if name == 'noise' 
                         else self.config.kl_weight)
                kl_total += weight * kl_loss
            
            losses['kl_total'] = kl_total
            total_loss += kl_total
        
        # 3. Classification losses
        if self.config.classification:
            for name, enc_output in encoded.items():
                if 'logits' in enc_output and name in labels:
                    cls_loss = self.classification_loss(enc_output['logits'], labels[name])
                    losses[f'classification_{name}'] = cls_loss
                    total_loss += self.config.classification_weight * cls_loss
        
        # 4. Self-generation loss (reconstruct from sampled z)
        if self.config.self_generation:
            # Sample new z from learned distribution
            sampled_codes = {}
            for name, enc_output in encoded.items():
                mu, logvar = enc_output['mu'], enc_output['logvar']
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                sampled_codes[name] = mu + eps * std
            
            gen_reconstruction = model.generator(sampled_codes)
            gen_loss = F.mse_loss(gen_reconstruction, features) # TODO which one? 
            #gen_loss = self.self_reconstruction_loss(gen_reconstruction, labram_features)
            losses['self_generation'] = gen_loss
            total_loss += self.config.self_generation_weight * gen_loss
        
        # 5. Latent consistency loss
        if self.config.latent_consistency:
            # Re-encode reconstruction to check consistency
            reencoded = model.encode(reconstruction.detach())
            consistency_loss = self.latent_consistency_loss(encoded, reencoded)
            losses['latent_consistency'] = consistency_loss
            total_loss += self.config.latent_consistency_weight * consistency_loss
        
        # 6. Self-cycle consistency (encode->decode->encode)
        if self.config.self_cycle:
            cycle_reconstruction = model.decode(encoded)
            cycle_reencoded = model.encode(cycle_reconstruction)
            cycle_loss = 0
            for name in encoded.keys():
                mu_loss = F.mse_loss(encoded[name]['mu'], cycle_reencoded[name]['mu'])
                logvar_loss = F.mse_loss(encoded[name]['logvar'], cycle_reencoded[name]['logvar'])
                cycle_loss += mu_loss + logvar_loss
            losses['self_cycle'] = cycle_loss / len(encoded)
            total_loss += self.config.self_cycle_weight * cycle_loss
            
        # 7. Cross-subject intra-class / cross-class generation
        if self.config.cross_subject_intra_class:
            intra_recon = outputs['cross_intra_reconstruction']
            intra_target = outputs['cross_intra_target']
            intra_loss = self.cross_subject_intra_class_loss(intra_recon, intra_target)
            losses['cross_subject_intra_class'] = intra_loss
            total_loss += self.config.cross_subject_intra_class_weight * intra_loss

        if self.config.cross_subject_cross_class:
            cross_recon = outputs['cross_cross_reconstruction']
            cross_target = outputs['cross_cross_target']
            cross_loss = self.cross_subject_cross_class_loss(cross_recon, cross_target)
            losses['cross_subject_cross_class'] = cross_loss
            total_loss += self.config.cross_subject_cross_class_weight * cross_loss
            
            

            
        # 8. Adversarial losses (WGAN-GP)
        # -------------------------------------------------------------------------
        if hasattr(self.config, 'adversarial') and self.config.adversarial:
            real_data = features.detach()
            fake_data = reconstruction.detach()
            D_real = model.discriminator(real_data)
            D_fake = model.discriminator(fake_data)

            adv_gen_loss = -torch.mean(D_fake)
            adv_dis_loss = torch.mean(D_fake) - torch.mean(D_real)

            # Gradient Penalty
            alpha = torch.rand(real_data.size(0), 1, 1, 1, device=outputs['reconstruction'].device)
            interpolated = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
            D_interpolated = model.discriminator(interpolated)
            grad_outputs = torch.ones_like(D_interpolated)
            gradients = grad(outputs=D_interpolated, inputs=interpolated,
                            grad_outputs=grad_outputs, create_graph=True,
                            retain_graph=True, only_inputs=True)[0]
            grad_penalty = ((gradients.view(gradients.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()

            losses['gradient_penalty'] = grad_penalty
            losses['adversarial_generator'] = adv_gen_loss
            losses['adversarial_discriminator'] = adv_dis_loss

            total_loss += self.config.lambda_gp * grad_penalty
            total_loss += self.config.adversarial_weight * adv_gen_loss
            
        # 9. Adaptive loss balancing (Eq. 16)
        if self.config.adaptive_balancing and 'prev_losses' in self.state:
            # Compute rate of change between iterations
            for key in ['self_reconstruction', 'kl_total', 'classification_task']:
                if key in losses and key in self.state['prev_losses']:
                    rate = abs(losses[key] - self.state['prev_losses'][key]) / (
                        abs(self.state['prev_losses'][key]) + 1e-8
                    )
                    alpha = torch.clamp(rate, 0.5, 2.0)  # scaling range
                    losses[key] = alpha * losses[key]

            # Update previous loss states
            self.state['prev_losses'].update({k: v.detach() for k, v in losses.items() if torch.is_tensor(v)})

                
        # -------------------------------------------------------------------------
        losses['total'] = total_loss
        return losses

    def compute_loss_classification_only(self, outputs: Dict[str, Any], labels: Dict[str, torch.Tensor],
                    model: SimpleFeaturesClassifier) -> Dict[str, torch.Tensor]:
        """Compute all enabled losses."""
        losses = {}
        total_loss = torch.tensor(0.0, device='cuda:0')
        

        logits = outputs

        # 1. Classification loss
        if self.config.classification:
            cls_loss = F.cross_entropy(logits, labels['task'])
            losses['classification'] = cls_loss
            total_loss += self.config.classification_weight * cls_loss
        losses['total'] = total_loss
        return losses

