import braindecode

# import labram model
from braindecode.models import Labram
import torch
import torch.nn as nn
from tqdm import tqdm

from CBraMod_main.models import cbramod
from CBraMod_main.models.cbramod import CBraMod
from einops.layers.torch import Rearrange

def create_labram_model(input_shape, n_classes, patch_size):
    """
    Create a Labram model for EEG signal classification.

    Parameters:
    input_shape (tuple): Shape of the input data (channels, timepoints).
    n_classes (int): Number of output classes.
    patch_size (int): Size of patches for the model.

    Returns:
    model: An instance of the Labram model.
    """
    model = Labram(
        n_chans=input_shape[0],
        n_times=input_shape[1],
        n_outputs=n_classes,
        patch_size=patch_size,
        emb_size=200,
    )
    return model


def create_cbramod_model(input_shape, emb_dim, patch_size=None):
    """
    Create a CBRAModel for EEG signal classification.

    Parameters:
    input_shape (tuple): Shape of the input data (channels, timepoints).
    emb_dim (int): Dimension of the embedding output.

    Returns:
    model: An instance of the CBRAModel.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    patch_num = input_shape[1] // patch_size if patch_size is not None else 30 #TODO which default?
    model = CBraMod(in_dim=input_shape[0], out_dim=emb_dim, d_model=emb_dim, seq_len=patch_num).to(device)
    model.load_state_dict(torch.load('weights/cbramod_pretrained_weights.pth', map_location=device))
    model.proj_out = nn.Identity()
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)  # freeze

    return model

def segment_to_patches(x, patch_size):
    """Convert (B, C, T) -> (B, C, n_patches, patch_size). Trims tail if needed."""
    B, C, T = x.shape
    n_patches = T // patch_size
    if n_patches == 0:
        raise ValueError(f"patch_size {patch_size} larger than signal length {T}")
    x = x[:, :, : n_patches * patch_size]  # trim remainder (or pad if you prefer)
    return x.view(B, C, n_patches, patch_size).contiguous()


WHICH_DATA = "MI_eeg"  # options: "MI_eeg", "sleepedfx"
#WHICH_DATA = "sleepedfx"
#WHICH_DATA = "erp"

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
    potential_sizes = []
    for n_patches in range(min_patches, max_patches + 1):
        patch_size = signal_length // n_patches
        # Check if this patch size divides signal length somewhat evenly
        if signal_length % patch_size < patch_size * 0.2:  # Allow 20% of patch size as remainder
            potential_sizes.append((patch_size, n_patches))
    
    if not potential_sizes:
        return signal_length // 8  # fallback to 8 patches if no good divisors found
    
    # Choose the patch size that gives us closest to 8 patches (reasonable middle ground)
    return min(potential_sizes, key=lambda x: abs(x[1] - 8))[0]



# Set appropriate patch sizes for each dataset
if WHICH_DATA == 'sleepedfx':
    time_samples = 240
    patch_size = 60  # divides evenly into 240
elif WHICH_DATA == 'MI_eeg':
    time_samples = 481
    patch_size = get_optimal_patch_size(time_samples)
    print(f"Optimal patch size for MI data: {patch_size}")
    n_patches = time_samples // patch_size
    print(f"Using patch size {patch_size} for MI data ({n_patches} patches)")
    #exit() #! lia added #TODO hier weiter machen !
elif WHICH_DATA == 'erp':
    time_samples = 256
    patch_size = get_optimal_patch_size(time_samples)
    n_patches = time_samples // patch_size
    print(f"Using patch size {patch_size} for ERP data ({n_patches} patches)")
else:
    raise ValueError(f"Unknown dataset: {WHICH_DATA}")

MODEL = "labram"  # options: "labram", "cbramod"
MODEL = "cbramod"
if MODEL == "labram":
    print("Using Labram model")
    save_name = f"{WHICH_DATA}_patch{patch_size}_{MODEL}_features.pt"
    model = create_labram_model(input_shape=(64, time_samples), n_classes=0, patch_size=patch_size)
elif MODEL == "cbramod":
    print("Using CBRAModel")
    save_name = f"{WHICH_DATA}_{MODEL}_features.pt"
    model = create_cbramod_model(input_shape=(200, time_samples), emb_dim=200, patch_size=patch_size)
#print(weights['model'].keys())



tokenizer_weights = torch.load('weights/vqnsp.pth', map_location='cpu', weights_only=False)
#~ print(tokenizer_weights['model'].keys())
# model_weights = torch.load('weights/labram-base.pth', map_location='cpu', weights_only=False)
# print(model_weights['model'].keys())

# Load pretrained weights into the model
if MODEL == "labram":
    print("Loading Labram pretrained weights")
    model.load_state_dict(tokenizer_weights['model'], strict=False)

if WHICH_DATA == 'sleepedfx':
    print("Processing Sleep-EDF Expanded data")
    data = torch.load("data/processed_data/sleepedfx_data.pt", weights_only=False)
    print("keys:", data.keys())
    inputs = data['data']  # shape (n_samples, n_channels, n_times)
    print(f"data loaded: {inputs.shape}, {data['labels'].shape}")
    
    # divide in batches
    batch_size = 32
    n_batches = (inputs.shape[0] // batch_size) + 1

    ptr = 0
    res_all = []
    for i in tqdm(range(n_batches)):
        ptr = i * batch_size
        if ptr >= inputs.shape[0]:
            break
        if i == n_batches - 1: # take all remaining samples
            batch = inputs[ptr:]
        else:
            batch = inputs[ptr:ptr + batch_size]
        
        # normalize batch
        batch = (batch - batch.mean(dim=2, keepdim=True)) / (batch.std(dim=2, keepdim=True) + 1e-6)
        # pass normalized batch through model
        if MODEL == "labram":
            res = model.forward_features(batch)
        elif MODEL == "cbramod":
            res = model.forward(batch)
        res_all.extend(res.detach().cpu().numpy())

    # save features, assert that the total number of features is equal to the number of samples

    res_all = torch.tensor(res_all)
    assert res_all.shape[0] == inputs.shape[0]

    to_save = {'features': res_all, 'labels': data['labels'], 'subjects': data['subjects'], 'runs': data['runs'], 'tasks': data['tasks']}
    # use save_name defined earlier
    torch.save(to_save, f"data/processed_data/{save_name}")
    


if WHICH_DATA == 'MI_eeg':    
    # Motor imagery data
    data = torch.load("data/processed_data/MI_eeg.pt")
    print(f"data loaded: {data['X'].shape}, {data['y'].shape}")
    inputs = data['X']

    # divide in batches
    batch_size = 32
    n_batches = (inputs.shape[0] // batch_size) + 1
    print(f"Number of batches: {n_batches}")

    ptr = 0
    res_all = []
    res_list = []
    total_collected = 0
    for i in tqdm(range(n_batches)):
        ptr = i * batch_size
        if ptr >= inputs.shape[0]:
            break
        if i == n_batches - 1: # take all remaining samples
            batch = inputs[ptr:]
            print(f"Last batch size: {batch.shape}")
        else:
            batch = inputs[ptr:ptr + batch_size]
            #print(f"Batch size: {batch.shape}")
        
        # normalize batch
        batch = (batch - batch.mean(dim=2, keepdim=True)) / (batch.std(dim=2, keepdim=True) + 1e-6)
        #print batch shape
        #print(f"Normalized batch shape: {batch.shape}")
        # pass normalized batch through model
        if MODEL == "labram":
            res = model.forward_features(batch)
        elif MODEL == "cbramod":
            # convert (B, C, T) -> (B, C, n_patches, patch_size)
            device = next(model.parameters()).device
            batch_patched = segment_to_patches(batch, patch_size).to(device)
            res = model.forward(batch_patched)  
            
        #print(f"Feature shape: {res.shape}")
        #res_all.extend(res.detach().cpu().numpy())
        res_list.append(res.detach().cpu())
        total_collected += res.shape[0]
        
        print(f"Total features collected so far: {total_collected}")
    # save features, assert that the total number of features is equal to the number of samples
    # single concatenation (fast)
    res_all = torch.cat(res_list, dim=0)
    print(f"Total features shape: {res_all.shape}")

    #res_all = torch.tensor(res_all)
    print(f"Total features shape: {res_all.shape}")
    assert res_all.shape[0] == inputs.shape[0]
    
    if MODEL == "cbramod":
        # average over channels and patches
        res_all = res_all.mean(dim=(1, 2))  

    to_save = {'features': res_all, 'labels': data['y'], 'subjects': data['subjects'], 'runs': data['runs']}
    print(f"Saving features to data/processed_data/{save_name}")
    torch.save(to_save, f"data/processed_data/{save_name}")
    #torch.save(to_save, "data/processed_data/MI_eeg_labram_features.pt")
    
    
if WHICH_DATA == 'erp':    
    # Motor imagery data
    data = torch.load("data/processed_data/full_erp_data.pt")
    print(f"data loaded: {data['X'].shape}, {data['y'].shape}")
    inputs = data['X']

    # divide in batches
    batch_size = 32
    n_batches = (inputs.shape[0] // batch_size) + 1

    ptr = 0
    res_all = []
    for i in tqdm(range(n_batches)):
        ptr = i * batch_size
        if ptr >= inputs.shape[0]:
            break
        if i == n_batches - 1: # take all remaining samples
            batch = inputs[ptr:]
        else:
            batch = inputs[ptr:ptr + batch_size]
        
        # normalize batch
        batch = (batch - batch.mean(dim=2, keepdim=True)) / (batch.std(dim=2, keepdim=True) + 1e-6)
        # pass normalized batch through model
        if MODEL == "labram":
            res = model.forward_features(batch)
        elif MODEL == "cbramod":
            res = model.forward(batch)
        res_all.extend(res.detach().cpu().numpy())

    # save features, assert that the total number of features is equal to the number of samples

    res_all = torch.tensor(res_all)
    assert res_all.shape[0] == inputs.shape[0]

    to_save = {'features': res_all, 'labels': data['y'], 'subjects': data['subjects'], 'runs': data['runs']}
    torch.save(to_save, f"data/processed_data/{save_name}")
    #torch.save(to_save, "data/processed_data/MI_eeg_labram_features.pt")
    
    