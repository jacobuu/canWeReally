import argparse
import logging
from functools import partial
from pathlib import Path
from typing import List, Optional

import librosa
import mne
import numpy as np
import torch
from scipy import signal
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

logger = logging.getLogger(__name__)



def process_sleepedfx_data(args):
    N_SUBJECTS = args.subjects
    SAMPLING_RATE = args.fs
    DATA_DIR = Path(args.data)
    OUTPUT_DIR = Path(args.output)
    FILE_NAME = "sleepedfx_data.pt" if args.output_file is None else args.output_file
    print(f"Processing SleepEDFx data from {DATA_DIR} and saving to {OUTPUT_DIR / FILE_NAME}")
 


    filelist = mne.datasets.sleep_physionet.age.fetch_data(
        subjects=list(range(N_SUBJECTS)), path=DATA_DIR, on_missing="warn"
    )
    annotation_desc_2_event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3": 4,
        "Sleep stage 4": 4,
        "Sleep stage R": 5,
    }
    # create a new event_id that unifies stages 3 and 4
    event_id = {
        "Sleep stage W": 1,
        "Sleep stage 1": 2,
        "Sleep stage 2": 3,
        "Sleep stage 3/4": 4,
        "Sleep stage R": 5,
    }
    Zxx_array = None
    subjects = []
    nights = []
    tasks = []
    for idx, file_pair in enumerate(tqdm(filelist)):
        edf_path = Path(file_pair[0])
        hyp_path = Path(file_pair[1])

        raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel="Event marker", infer_types=True)
        annot = mne.read_annotations(hyp_path)
        raw.set_annotations(annot, emit_warning=False)

        # keep last 30-min wake events before sleep and first 30-min wake events after
        # sleep and redefine annotations on raw data
        first_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][0]
        last_wake = [i for i, x in enumerate(annot.description) if x == "Sleep stage W"][-1]
        if first_wake is not None and last_wake is not None:
            annot.crop(annot[first_wake + 1]["onset"] - 30 * 60, annot[last_wake]["onset"] + 30 * 60)
        else:
            ...
            # continue
        raw.set_annotations(annot, emit_warning=False)
        events, _ = mne.events_from_annotations(raw, event_id=annotation_desc_2_event_id, chunk_duration=30.0)

        
        # Resample data and events
        print(raw.info['sfreq'])
        logger.info(f"Resampling data from {raw.info['sfreq']} to {SAMPLING_RATE} Hz")
        raw.resample(SAMPLING_RATE, events=events, npad="auto")
        
        # Apply the necessary filters to the raw continuous data
        raw.filter(l_freq=0.3, h_freq=75.0, fir_design='firwin', verbose=False)
        raw.notch_filter(freqs=60.0, fir_design='firwin', verbose=False)


        tmax = 30.0 - 1.0 / SAMPLING_RATE
        epochs = mne.Epochs(
            raw=raw, events=events, event_id=event_id, tmin=0.0, tmax=tmax, baseline=None, preload=True, on_missing="warn"
        )
        
        epochs.pick_types(eeg=True, exclude='bads') 
        X = epochs.get_data() # X now has shape (N_epochs, N_channels, N_samples)

        cbramod_norm = True  # TODO: Set to True to use CBRAMod normalization, False for Z-score normalization
        if cbramod_norm:
            # We normalize EEG by setting the unit to 100 µV to guarantee the value mainly between -1 to 1.
            scaling_factor = 100.0  
            X = X / scaling_factor
        else: # Z-score normalization
            # Calculate global mean/std for normalization on RAW data
            mu = X.mean()
            std = X.std()

            # Normalize data (this is the RAW EEG segment)
            X = (X - mu) / std

        # Set Zxx_array (the final output) to the raw data X
        Zxx = X 

        # Update N, C, T dimensions from the raw EEG data
        N, C, T = Zxx.shape

        # Append to array (must handle multi-dimensional array concatenation)
        if Zxx_array is None:
            Zxx_array = Zxx
        else:
            # Concatenate along the N_epochs axis (axis=0)
            Zxx_array = np.concatenate((Zxx_array, Zxx), axis=0)
    
        # Save metadata
        N, F, T = Zxx.shape
        subjects.extend([int(edf_path.stem[3:5])] * N)
        nights.extend([int(edf_path.stem[5:6])] * N)
        tasks.extend(epochs.events[:, -1].tolist())

    Zxx_array = torch.from_numpy(Zxx_array).float()

    # Do splits and calculate split mean and std
    splits = list(GroupKFold(n_splits=5).split(X=Zxx_array, y=tasks, groups=subjects))
    fold_info = {i: {"train_idx": None, "test_idx": None, "mean": None, "std": None} for i, _ in enumerate(splits)}
    for i, (train_index, test_index) in enumerate(splits):
        fold_info[i]["train_idx"] = train_index
        fold_info[i]["test_idx"] = test_index
        fold_info[i]["mean"] = torch.mean(Zxx_array[train_index])
        fold_info[i]["std"] = torch.std(Zxx_array[train_index])
    torch.save(
        dict(
            data=Zxx_array,
            subjects=torch.tensor(subjects),
            tasks=torch.tensor(tasks),
            runs=torch.tensor(nights),
            labels=torch.tensor(tasks),
            fold_info=fold_info,
        ),
        OUTPUT_DIR / FILE_NAME,
    )

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="SleepEDFx data processing")
    parser.add_argument('-d', '--data', type=str, help='Data directory path')
    parser.add_argument('-o', '--output', type=str, help='Output file path')
    parser.add_argument('-n', '--subjects', type=int, default=83, help='Number of subjects to process')
    parser.add_argument('--fs', type=int, default=200, help='Sampling frequency')
    parser.add_argument('--output-file', type=str, default=None, help='Output file name')
    args = parser.parse_args()

    process_sleepedfx_data(args)
    
    """Example usage:
    python data/data_preparation/create_sleepedfx_cbramod.py -d /var/datasets/physionet.org/files/sleep-edfx/1.0.0/sleep-cassette -o /home/burger/canWeReally/data/processed_data --fs 200 --output-file sleepedfx_cbramod_data.pt
    """