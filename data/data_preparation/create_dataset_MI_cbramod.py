import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interattivo per SSH
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import mne

# ============================================================================
# CARICAMENTO DATI
# ============================================================================


SAMPLING_RATE = 200 


selected_channels = ['Fc5.', 'Fc3.', 'Fc1.', 'Fcz.', 'Fc2.', 'Fc4.', 'Fc6.', 'C5..', 'C3..', 'C1..', 'Cz..', 'C2..',
                     'C4..', 'C6..', 'Cp5.', 'Cp3.', 'Cp1.', 'Cpz.', 'Cp2.', 'Cp4.', 'Cp6.', 'Fp1.', 'Fpz.', 'Fp2.',
                     'Af7.', 'Af3.', 'Afz.', 'Af4.', 'Af8.', 'F7..', 'F5..', 'F3..', 'F1..', 'Fz..', 'F2..', 'F4..',
                     'F6..', 'F8..', 'Ft7.', 'Ft8.', 'T7..', 'T8..', 'T9..', 'T10.', 'Tp7.', 'Tp8.', 'P7..', 'P5..',
                     'P3..', 'P1..', 'Pz..', 'P2..', 'P4..', 'P6..', 'P8..', 'Po7.', 'Po3.', 'Poz.', 'Po4.', 'Po8.',
                     'O1..', 'Oz..', 'O2..', 'Iz..']


def load_data(file_path, tmin=-1, tmax=6):
    """
    Carica e preprocessa dati EEG
    """
    raw = mne.io.read_raw_edf(file_path, preload=True, 
                               stim_channel='Event marker', verbose=False)
    raw.pick_channels(selected_channels, ordered=True) 
    if len(raw.info['bads']) > 0:
        print('interpolate_bads')
        raw.interpolate_bads()
    run = file_path.split('/')[-1][5:7]
    print("file_path: ", file_path)
    run = int(run) if run.isdigit() else None
    print("run: ", run)
    sfreq = raw.info['sfreq']
    raw.set_eeg_reference('average')

    raw.filter(l_freq=0.3, h_freq=None)
    raw.notch_filter((60))

    raw.resample(SAMPLING_RATE)
    
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    
    
     
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=0, tmax=4. - 1.0 /SAMPLING_RATE, 
                        baseline=None, preload=True, verbose=False)
    
    # remove the . from the channel names
    channels = [ch.replace('.', '') for ch in raw.info['ch_names']]
    epochs.rename_channels({old: new for old, new in zip(epochs.info['ch_names'], channels)})

    # --- mappa T0/T1/T2 in base al tipo di run ---
    if run in [4, 8, 12]:
        # left vs right fist
        label_map = {'T0': 4, 'T1': 0, 'T2': 1}
    elif run in [6, 10, 14]:
        # both fists vs both feet
        label_map = {'T0': 4, 'T1': 2, 'T2': 3}
    else:
        return None, None, None, None, None
    
       

    
    data = []
    labels = []
    for code, label in label_map.items():
        if code in epochs.event_id:
            ep_data = epochs[code].get_data(units='uV')
            # if ep_data.shape[2] != int((tmax - tmin) * SAMPLING_RATE) + 1:
            #     print(f"Skipping {file_path} due to unexpected shape")
            #     continue
            data.append(ep_data)
            labels.extend([label] * len(ep_data))
    
    if len(data) == 0:
        return None, None, None, None, None

    runs = [run] * len(labels)
    subjects = [int(file_path.split('/')[-2][1:])] * len(labels)
    return np.concatenate(data), np.array(labels), epochs.ch_names, runs, subjects


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("CSP Analysis - Implementazione Completa")
    print("="*70)
    
    # File di input
    # sample_file = "/var/datasets/EEG_MI/files/eegmmidb/1.0.0/S001/S001R04.edf"
    
    print("\n1. Caricamento dati...")
    # load all data from subject S001, runs 3 to 14
    all_signals = []
    all_labels = []
    all_ch_names = None
    all_runs = []
    all_subjects = []
    dataset_path = "/var/datasets/physionet.org/files/MI/files/eegmmidb/1.0.0/"
    for subj in range(1, 110):
        for run in range(3, 15):
            file_path = f"{dataset_path}S{subj:03d}/S{subj:03d}R{run:02d}.edf"
            print(f"   Caricamento {file_path}...")
            X_run, y_run, ch_names, runs, subjects = load_data(file_path, tmin=1, tmax=4)
            if X_run is not None and y_run is not None:
                all_signals.extend(X_run)
                all_labels.extend(y_run)
                all_runs.extend(runs)
                all_subjects.extend(subjects)
                if all_ch_names is None:
                    all_ch_names = ch_names
    X = torch.tensor(np.array(all_signals), dtype=torch.float32)
    y = torch.tensor(np.array(all_labels), dtype=torch.int64)
    runs = torch.tensor(np.array(all_runs), dtype=torch.int64)
    subjects = torch.tensor(np.array(all_subjects), dtype=torch.int64)
    dizionario = {'X': X, 'y': y, 'ch_names': all_ch_names, 'runs': runs, 'subjects': subjects}

    torch.save(dizionario, "../processed_data/MI_eeg_cbramod.pt")
    print("Dati salvati in MI_eeg_cbramod.pt")
