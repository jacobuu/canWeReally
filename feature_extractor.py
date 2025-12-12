import torch
import torch.nn as nn
from tqdm import tqdm
from braindecode.models import Labram
from CBraMod_main.models.cbramod import CBraMod

def create_labram_model(input_shape, n_classes, patch_size, emb_size=200):
    """
    Create a Labram model for EEG signal classification.

    Parameters:
    input_shape (tuple): Shape of the input data (channels, timepoints).
    n_classes (int): Number of output classes.
    patch_size (int): Size of patches for the model.

    Returns:
    model: An instance of the Labram model.
    """
    return Labram(n_chans=input_shape[0], n_times=input_shape[1], n_outputs=n_classes,
                  patch_size=patch_size, emb_size=emb_size)

def create_cbramod_model(in_dim, time_samples, emb_dim=200, patch_size=200, weights_path='weights/cbramod_pretrained_weights.pth'):
    """
    Create a CBRAModel for EEG signal classification.

    Parameters:
    in_dim (tuple): Shape of the input data (channels, timepoints).
    time_samples (int): Number of time samples in the input data.
    emb_dim (int): Dimension of the embedding output.

    Returns:
    model: An instance of the CBRAModel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seq_len = time_samples // patch_size
    model = CBraMod(in_dim=in_dim, out_dim=emb_dim, d_model=emb_dim, seq_len=seq_len).to(device)
    weights = torch.load(weights_path, map_location=device)
    if any('backbone.' in k for k in weights.keys()):
        # if weights were saved with a backbone prefix then strip it
        weights = {k.replace('backbone.', ''): v for k, v in weights.items()}
        # remove also the classifier. head if present
        weights = {k: v for k, v in weights.items() if not k.startswith('classifier.')}
    model.load_state_dict(weights, strict=False)
    model.proj_out = nn.Identity()
    model.eval()
    for p in model.parameters(): 
        p.requires_grad_(False)
    return model

def segment_to_patches(x, patch_size):
    """Convert (B, C, T) -> (B, C, n_patches, patch_size). Trims tail if needed."""
    B, C, T = x.shape
    n_patches = T // patch_size
    if n_patches == 0:
        raise ValueError(f"patch_size {patch_size} larger than signal length {T}")
    x = x[:, :, : n_patches * patch_size]
    return x.view(B, C, n_patches, patch_size).contiguous()

def get_optimal_patch_size(signal_length, min_patches=4, max_patches=16):
    """
    Calculate optimal patch size based on signal length.
    
    Parameters:
    signal_length (int): Length of the input signal
    min_patches (int): Minimum number of patches desired
    max_patches (int): Maximum number of patches desired
    
    Returns:
    int: Optimal patch size
    """
    candidates = []
    for n_patches in range(min_patches, max_patches + 1):
        patch_size = signal_length // n_patches
        if signal_length % patch_size < 0.2 * patch_size:
            candidates.append((patch_size, n_patches))
    if not candidates:
        return signal_length // 8   # fallback to 8 patches if no good divisors found
    # Choose the patch size that gives us closest to 8 patches (reasonable middle ground)
    return min(candidates, key=lambda x: abs(x[1] - 8))[0]

def extract_features_and_save(inputs, model, model_type, patch_size, save_path, batch_size=32, labels=None, extra_meta=None):
    """
    Extract features from inputs using the specified model and save to disk.
    
    Parameters:
    inputs (torch.Tensor): Input data of shape (B, C, T).
    model: The feature extraction model.
    model_type (str): Type of the model ('labram' or 'cbramod').
    patch_size (int): Patch size for models that require it.
    save_path (str): Path to save the extracted features.
    labels (torch.Tensor, optional): Labels corresponding to inputs.
    extra_meta (dict, optional): Additional metadata to save. Like subject IDs, run IDs, task etc.
    
    Returns:
    None"""
    device = next(model.parameters()).device
    batch_size = min(batch_size, inputs.shape[0])
    n_batches = (inputs.shape[0] // batch_size) + 1
    res_list = []
    total_collected = 0
    for i in tqdm(range(n_batches)):
        ptr = i * batch_size
        if ptr >= inputs.shape[0]:
            break
        batch = inputs[ptr:] if i == n_batches - 1 else inputs[ptr:ptr + batch_size]
        # normalize
        batch = (batch - batch.mean(dim=2, keepdim=True)) / (batch.std(dim=2, keepdim=True) + 1e-6)
        
        if model_type == "labram":
            out = model.forward_features(batch.to(device))
        elif model_type == "cbramod":
            # cbramod expects (B, C, n_patches, patch_size). convert (B, C, T) -> (B, C, n_patches, patch_size)
            patched = segment_to_patches(batch, patch_size).to(device)
            #print("patched.shape:", patched.shape)                       # (B, C, n_patches, patch_size_you_created)
            out = model.forward(patched)
            
        res_list.append(out.detach().cpu())
        total_collected += out.shape[0]
        
    features = torch.cat(res_list, dim=0)
    assert features.shape[0] == inputs.shape[0], f"collected {features.shape[0]} vs expected {inputs.shape[0]}"
    
    if model_type == "cbramod":
        features = features.mean(dim=(1, 2))
        
    to_save = {'features': features}
    
    if labels is not None:
        to_save['labels'] = labels
    # also include subject/run/task info if available

    if extra_meta:
        to_save.update(extra_meta)
    torch.save(to_save, save_path)
    print(f"Saved features to {save_path}")


#------------------------------------------------------------------
# configuration
WHICH_DATA = "MI_eeg"  # options: MI_eeg, sleepedfx, erp
MODEL = "cbramod"      # options: labram, cbramod
#------------------------------------------------------------------

# dataset-specific settings
if WHICH_DATA == 'sleepedfx':
    time_samples = 240
    patch_size = 60
elif WHICH_DATA == 'MI_eeg':
    time_samples = 800
    patch_size = 200 # get_optimal_patch_size(time_samples)
elif WHICH_DATA == 'erp':
    time_samples = 256
    patch_size = 200# get_optimal_patch_size(time_samples)
else:
    raise ValueError(f"Unknown dataset {WHICH_DATA}")

# prepare model (for cbramod we delay instantiation until we know channels for the dataset)
model = None
save_name = f"{WHICH_DATA}_{MODEL}_finetunedAll_patch{patch_size}_features_TEST.pt"

# load tokenizer/other weights if needed (kept for compatibility)
try:
    tokenizer_weights = torch.load('weights/vqnsp.pth', map_location='cpu', weights_only=False)
except Exception:
    tokenizer_weights = None

if MODEL == "labram":
    # instantiate immediately (channels known from config)
    model = create_labram_model(input_shape=(64, time_samples), n_classes=0, patch_size=patch_size)
    if tokenizer_weights:
        model.load_state_dict(tokenizer_weights['model'], strict=False)

# dataset processing
if WHICH_DATA == 'MI_eeg':
    data = torch.load("data/processed_data/MI_eeg.pt")
    inputs = data['X']
    labels = data.get('y', None)
    # ensure model exists and matches channels
    if MODEL == "cbramod":
        n_channels = inputs.shape[1]
        # model = create_cbramod_model(in_dim=patch_size, time_samples=time_samples, emb_dim=200, patch_size=patch_size)
        model = create_cbramod_model(in_dim=patch_size, time_samples=time_samples, emb_dim=200, patch_size=patch_size, weights_path='/home/burger/canWeReally/weights/finetuned/epoch40_acc_0.62291_kappa_0.49715_f1_0.62318.pth')
        print("Created cbramod model for MI_eeg with", n_channels, "channels")
    extract_features_and_save(inputs, model, MODEL, patch_size, f"data/processed_data/{save_name}",
                              labels=labels, extra_meta={'subjects': data.get('subjects'), 'runs': data.get('runs')})

elif WHICH_DATA == 'sleepedfx':
    data = torch.load("data/processed_data/sleepedfx_data.pt", weights_only=False)
    inputs = data['data']
    labels = data['labels']
    if MODEL == "cbramod":
        n_channels = inputs.shape[1]
        model = create_cbramod_model(in_dim=200, time_samples=time_samples, emb_dim=200, patch_size=patch_size)
    extract_features_and_save(inputs, model, MODEL, patch_size, f"data/processed_data/hehehe{save_name}",
                              labels=labels, extra_meta={'subjects': data.get('subjects'), 'runs': data.get('runs'), 'tasks': data.get('tasks')})

elif WHICH_DATA == 'erp':
    data = torch.load("data/processed_data/full_erp_data.pt")
    inputs = data['X']
    labels = data['y']
    if MODEL == "cbramod":
        n_channels = inputs.shape[1]
        model = create_cbramod_model(in_dim=200, time_samples=time_samples, emb_dim=200, patch_size=patch_size)
    extract_features_and_save(inputs, model, MODEL, patch_size, f"data/processed_data/{save_name}",
                              labels=labels, extra_meta={'subjects': data.get('subjects'), 'runs': data.get('runs')})