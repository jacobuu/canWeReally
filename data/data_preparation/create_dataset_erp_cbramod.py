import mne
mne.set_log_level('critical')
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import sys


def parse_list(bdf_list):
    #Parse bdf_list to get capturing events and time window
    events = []
    time_window = None
    if 't<' in bdf_list:
        time_window = tuple(map(lambda x: float(x)/1000, bdf_list[bdf_list.find('<')+1:bdf_list.find('>')].split('-')))
        if '-' in bdf_list[bdf_list.find('>')+1:]:
            event_low, event_high = map(int, bdf_list[bdf_list.find('>')+1:bdf_list.find('}')].split('-'))
            events = range(event_low, event_high+1)
        else:
            events = map(lambda x: x.strip('{}'), bdf_list[bdf_list.find('>')+1:].split(';'))
    else:
        if '-' in bdf_list:
            event_low, event_high = map(int, bdf_list.strip('{}').split('-'))
            events = range(event_low, event_high+1)
        else:
            events = map(lambda x: x.strip('{}'), bdf_list.split(';'))
    return list(map(str, events)), time_window

def bin_annotations(annotations, bdf_format, bin_desc):
    #Bin annotations according to bdf_format
    pre_dot, post_dot = bdf_format.split('.')
    pre_events = []
    pre_time_window = None
    if pre_dot:
        pre_events, pre_time_window = parse_list(pre_dot)
    
    loc_dot = post_dot
    post_dot = post_dot.split('}{')
    suc_events = []
    suc_time_window = None
    if len(post_dot) > 1:
        suc_events, suc_time_window = parse_list(post_dot[1])
        loc_dot = post_dot[0]
    loc_events, _ = parse_list(loc_dot)
    
    locking_events = []
    locking_events_passed = []
    for i, annotation in enumerate(annotations):
        if annotation['description'] in loc_events:
            locking_events.append((i, annotation))
            locking_events_passed.append(False)

    for k, (i, ann) in enumerate(locking_events):
        onset = ann['onset']
        if pre_events and i > 0:
            pre_ann = annotations[i-1]
            if pre_ann['description'] in pre_events:
                if pre_time_window:
                    if onset - pre_ann['onset'] < pre_time_window[0] or onset - pre_ann['onset'] > pre_time_window[1]:
                        continue
            else:
                continue
        if suc_events and i < len(annotations)-1:
            suc_ann = annotations[i+1]
            if suc_ann['description'] in suc_events:
                if suc_time_window:
                    if suc_ann['onset'] - onset < suc_time_window[0] or suc_ann['onset'] - onset > suc_time_window[1]:
                        continue
            else:
                continue
        locking_events_passed[k] = True
    filtered_annotations = [ann for i, (_, ann) in enumerate(locking_events) if locking_events_passed[i]]   
    return mne.Annotations(onset=[ann['onset'] for ann in filtered_annotations], duration=0.0, description=bin_desc)


# ---------------------------------------------------------------------------------
# Core CBraMod Adaptation Function (Single Subject/Paradigm Processing)
# ---------------------------------------------------------------------------------

def process_erp_data(cur_dir, subj, para, full, event_ids_new, convert_dict_new, baseline, epoch_window, para_bins, para_labels, data, subjects, tasks, runs):
    
    # These are the two files used depending on whether full==True or full==False
    if not full:
        raw_name = f'{subj}/{subj}_{para}_shifted_ds_reref_ucbip_hpfilt_ica_prep1.set'
    else:
        raw_name = f'{subj}/{subj}_{para}_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set'
    
    # MMN data does not use a LCD-monitor and are therefore not time-shifted
    if para == 'MMN':
        raw_name = raw_name.replace('shifted_','')

    # Full are already stored as epochs
    if full:
        raw = mne.io.read_epochs_eeglab(cur_dir+raw_name)
    else: # Otherwise, read in the raw data
        raw = mne.io.read_raw_eeglab(cur_dir+raw_name, preload=True)

    # ----------------------------------------------------
    # CBraMod Adaptation: Filtering and Resampling (applied ONLY to raw data path)
    # ----------------------------------------------------
    if not full:
        target_fs = 200
        # 1. Resample to 200 Hz (Must be done first to allow 75 Hz filter)
        raw.resample(target_fs) 
        
        # 2. Band-pass filter: 0.3 Hz to 75 Hz 
        raw.filter(l_freq=0.3, h_freq=75.0, fir_design='firwin', verbose=False)
        
        # 3. Notch filter: 60 Hz 
        raw.notch_filter(freqs=60.0, fir_design='firwin', verbose=False)

    # The full data already has corrected EOG channels.
    if full:
        raw.set_channel_types({'(uncorr) HEOG':'eog', '(uncorr) VEOG':'eog',  '(corr) HEOG': 'emg', '(corr) VEOG': 'emg'})
    else:
        raw.set_channel_types({'(uncorr) HEOG':'eog', '(uncorr) VEOG':'eog'})
        
    # Dropping the uncorrected EOG channels/renaming
    if full:
        mne.rename_channels(raw.info, {'(corr) HEOG': 'HEOG','(corr) VEOG': 'VEOG'})
        raw.drop_channels(['(uncorr) HEOG', '(uncorr) VEOG'])
    else:
        mne.rename_channels(raw.info, {'(uncorr) HEOG': 'HEOG','(uncorr) VEOG': 'VEOG'})
        
    # These channels are used for re-referencing the corrected EOG channels, I drop them here.
    raw.drop_channels(['HEOG_left', 'HEOG_right', 'VEOG_lower'])

    # The full=False data is not epoched, so we need to epoch it
    if not full:
        if para == 'ERN':
            raw.set_annotations(bin_annotations(raw.annotations, '{t<200-1000>11;12;21;22}.{211;112;221;122}', 'Incorrect') + bin_annotations(raw.annotations, '{t<200-1000>11;12;21;22}.{111;212;121;222}', 'Correct'))
        elif para == 'LRP':
            raw.set_annotations(bin_annotations(raw.annotations, '{t<200-1000>21}.{121}', 'Contralateral') + bin_annotations(raw.annotations, '{t<200-1000>12}.{212}', 'Ipsilateral'))
        elif para == 'MMN':
            raw.set_annotations(bin_annotations(raw.annotations, '.{70}', 'Deviants') + bin_annotations(raw.annotations, '{80}.{80}', 'Standards'))
        elif para == 'N2pc':
            raw.set_annotations(bin_annotations(raw.annotations, '.{121;122}{t<200-1000>201}', 'Contralateral') + bin_annotations(raw.annotations, '.{211;212}{t<200-1000>201}', 'Ipsilateral'))
        elif para == 'N170':
            raw.set_annotations(bin_annotations(raw.annotations, '.{1-40}{t<200-1000>201}', 'Faces') + bin_annotations(raw.annotations, '.{41-80}{t<200-1000>201}', 'Cars'))
        elif para == 'N400':
            raw.set_annotations(bin_annotations(raw.annotations, '.{221;222}{t<200-1500>201}', 'Unrelated') + bin_annotations(raw.annotations, '.{211;212}{t<200-1500>201}', 'Related'))
        elif para == 'P3':
            raw.set_annotations(bin_annotations(raw.annotations, '.{11;22;33;44;55}{t<200-1000>201}', 'Rare') + bin_annotations(raw.annotations, '.{12;13;14;15;21;23;24;25;31;32;34;35;41;42;43;45;51;52;53;54}{t<200-1000>201}', 'Frequent'))

        events, event_dict = mne.events_from_annotations(raw, event_ids_new, regexp=None)
        raw = mne.Epochs(
            raw,
            events,
            event_dict,
            baseline=baseline[para],
            tmin=epoch_window[para][0],
            tmax=epoch_window[para][1],
            on_missing='ignore',
            preload=True,
        )
        # Crop to ~1 second after resampling to 200 Hz
        if para == 'LRP' or para == 'ERN':
            raw = raw.crop(tmin=raw.tmin,tmax=raw.times[-2])
        else:
            raw = raw.crop(tmin=raw.times[1],tmax=raw.tmax)
    else:
        # For the full data, we simply translate the defined bins to the new bin numbers.
        bins = para_bins[para]
        merged_event_dict = {convert_dict_new['1']: 1, convert_dict_new['2']: 2}
        old_bin_to_new = {}
        for n, b in raw.event_id.items():
            if str(bins[0]) in n[n.find('B')+1:n.find('(')]:
                old_bin_to_new[b] = 1
            elif str(bins[1]) in n[n.find('B')+1:n.find('(')]:
                old_bin_to_new[b] = 2
        raw.events[:,-1] = pd.Series(raw.events[:,-1]).map(old_bin_to_new).to_numpy()
        raw.event_id = merged_event_dict
        
    # ----------------------------------------------------
    # CBraMod Adaptation: Final Normalization (applied to both full=True/False paths)
    # ----------------------------------------------------
    
    # 1. Extract data (get_data returns Volts, V)
    data_epoch = raw.get_data(item=[convert_dict_new['1'], convert_dict_new['2']])
    
    # 2. Apply 100 µV fixed scaling factor: Normalize by setting unit to 100 µV.
    # 100 µV = 100 * 10^-6 V. Dividing by 100e-6 is equivalent to multiplying by 10,000.
    scaling_factor = 100e-6  
    data_epoch = data_epoch / scaling_factor
    
    # Adding the data to the lists
    data.append(data_epoch)
    subjects.append(np.array([subj]*data[-1].shape[0], dtype=int))
    tasks.append(pd.Series(raw.events[:,-1]).map(para_labels[para]).to_numpy())
    runs.append(np.array([para_num]*data[-1].shape[0], dtype=int))
    
    

if __name__ == '__main__':
    # ERP Core Bin-List Paradigm ids
    para_bins = {
        'ERN': [1, 2],
        'LRP': [4, 5],
        'MMN': [1, 2],
        'N2pc': [4, 5],
        'N170': [1, 2],
        'N400': [1, 2],
        'P3': [1, 2],
    }

    # Positive and negative bin id assignment
    para_labels = {
        'ERN': {1: 0, 2: 1},
        'LRP': {1: 2, 2: 3},
        'MMN': {1: 4, 2: 5},
        'N2pc': {1: 6, 2: 7},
        'N170': {1: 8, 2: 9},
        'N400': {1: 10, 2: 11},
        'P3': {1: 12, 2: 13},
    }

    # Paradigm/Event names
    para_label_names = {
        0: 'ERN/Incorrect',
        1: 'ERN/Correct',
        2: 'LRP/Contralateral',
        3: 'LRP/Ipsilateral',
        4: 'MMN/Deviants',
        5: 'MMN/Standards',
        6: 'N2pc/Contralateral',
        7: 'N2pc/Ipsilateral',
        8: 'N170/Faces',
        9: 'N170/Cars',
        10: 'N400/Unrelated',
        11: 'N400/Related',
        12: 'P3/Rare',
        13: 'P3/Frequent',
    }
    
    # Argument check for 'full' status - simplistic version
    # The user must provide the argument 'full' (e.g., python script.py full)
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'full':
        full = True
        output_file_tag = 'full_cbramod'
    else:
        full = False
        output_file_tag = 'simple_cbramod'
        
    # Lists for storing data
    data = []
    subjects = []
    tasks = []
    runs = []

    # Go through each paradigm
    for para_num, para in enumerate(tqdm(['ERN', 'LRP', 'MMN', 'N2pc', 'N170', 'N400', 'P3'])):
        # Folder where the paradigm data is stored
        cur_dir = f'/var/datasets/ERP-core/{para}/'
        
        # Baselines and epoch windows (unchanged)
        baseline = defaultdict(lambda: (-0.2,0))
        baseline['LRP'] = (-0.8,-0.6)
        baseline['ERN'] = (-0.4,-0.2)
        epoch_window = defaultdict(lambda: (-0.2, 0.8))
        epoch_window['LRP'] = (-0.8, 0.2)
        epoch_window['ERN'] = (-0.6, 0.4)
        
        # Positive and negative labels for each paradigm (unchanged)
        event_ids_new = {}
        convert_dict_new = {}
        if para == 'ERN':
            event_ids_new = {'Incorrect': 1, 'Correct': 2}
            convert_dict_new = {'1': 'Incorrect', '2': 'Correct'}
        elif para == 'LRP':
            event_ids_new = {'Contralateral': 1, 'Ipsilateral': 2}
            convert_dict_new = {'1': 'Contralateral', '2': 'Ipsilateral'}
        elif para == 'MMN':
            event_ids_new = {'Deviants': 1, 'Standards': 2}
            convert_dict_new = {'1': 'Deviants', '2': 'Standards'}
        elif para == 'N2pc':
            event_ids_new = {'Contralateral': 1, 'Ipsilateral': 2}
            convert_dict_new = {'1': 'Contralateral', '2': 'Ipsilateral'}
        elif para == 'N170':
            event_ids_new = {'Faces': 1, 'Cars': 2}
            convert_dict_new = {'1': 'Faces', '2': 'Cars'}
        elif para == 'N400':
            event_ids_new = {'Unrelated': 1, 'Related': 2}
            convert_dict_new = {'1': 'Unrelated', '2': 'Related'}
        elif para == 'P3':
            event_ids_new = {'Rare': 1, 'Frequent': 2}
            convert_dict_new = {'1': 'Rare', '2': 'Frequent'}
            
        # Go through each subject
        for subj in tqdm(range(1,41)):
            process_erp_data(cur_dir, subj, para, full, event_ids_new, convert_dict_new, baseline, epoch_window, para_bins, para_labels, data, subjects, tasks, runs)

    data = np.concatenate(data)
    subjects = np.concatenate(subjects)
    tasks = np.concatenate(tasks)
    runs = np.concatenate(runs)

    # REMOVE global mean/std calculation as CBraMod uses fixed scaling.

    # Saving with torch.save and loading with torch.load is faster than pickle.
    torch.save(dict(data=torch.from_numpy(data), subjects=torch.from_numpy(subjects), tasks=torch.from_numpy(tasks), runs=torch.from_numpy(runs), labels=para_label_names), f'./data/processed_data/{output_file_tag}_erp_data.pt')