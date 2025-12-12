import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from data.dataloaders.delegated_loader import DelegatedLoader
import torch

# CustomLoaderMI è ora un'istanza di torch.utils.data.Dataset
class CustomLoaderMI(Dataset):
    '''
    Custom data loader for MI data with filtering and indexing capabilities.
    Risolve il Data Leakage calcolando/applicando mean/std solo sul training set.
    '''
    
    def __init__(self, data_dict, split='train', data_mean=None, data_std=None, split_subjects=None, location=None):
        print("Initializing CustomLoaderMI...")
        # 1. Gestione Iniziale dei Dati
        if "data" not in data_dict:
            if "features" in data_dict:
                data_dict["data"] = data_dict.pop("features")
            elif "X" in data_dict:
                data_dict["data"] = data_dict.pop("X")
            else:
                raise KeyError("Data dictionary must contain 'data', 'features', or 'X' key")
        
        self.data_tensor = data_dict["data"]
        
        if "y" in data_dict:
            self.tasks = data_dict["y"].numpy()
        elif "labels" in data_dict:
            self.tasks = data_dict["labels"].numpy()
        else:
            raise KeyError("Data dictionary must contain 'y' or 'labels' key")
        
        # filter out invalid tasks (task 0) which corresponds to index 4. 
        valid_indices = np.where(self.tasks != 4)[0]
        self.data_tensor = self.data_tensor[valid_indices]
        self.tasks = self.tasks[valid_indices]
        self.subjects = data_dict["subjects"].numpy()[valid_indices]
        # Clamping runs to ensure they are >= 1 before converting to numpy
        self.runs = data_dict["runs"].clamp(min=1).numpy()[valid_indices]
        self.ch_names = data_dict["ch_names"] if "ch_names" in data_dict else None
        self.split = split
        self.split_subjects = split_subjects
        self.location = location
        
        # 2. Definizione dello Split
        train_ids = np.arange(70)    # Subjects 0-69 for training
        dev_ids = np.arange(70, 89)  # Subjects 70-88 for dev
        test_ids = np.arange(89, 109) # Subjects 89-108 for testing
        
        if split_subjects is not None:
            # Mode C: Use the provided list of subjects (e.g., T_A or T_V)
            self.unique_subjects_filter = split_subjects 
            print(f"CustomLoaderMI initialized for Mode C split based on {len(split_subjects)} provided subjects.")
        elif split == 'dev':
            self.unique_subjects_filter = dev_ids
        elif split == 'test':
            self.unique_subjects_filter = test_ids
        elif split == 'train':
            self.unique_subjects_filter = train_ids
        else:
            raise ValueError('Invalid split')
        
        print(f"CustomLoaderMI initialized for split: {split} with {len(self.unique_subjects_filter)} subjects.")
        
        
        # 3. Filtraggio Immediato dei Dati per lo Split Corrente
        data_indices = []
        for i, s in enumerate(self.subjects):
            if s in self.unique_subjects_filter:
                data_indices.append(i)
        
        # Copia dei dati filtrati sulla CPU e staccati dal computational graph
        self.data = self.data_tensor[data_indices].float().contiguous().detach().clone() / 100 # Normalize for cbramod
        
        # Copia dei metadati filtrati (uso np.ascontiguousarray per sicurezza)
        self.subjects = np.ascontiguousarray(self.subjects[data_indices])
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        # 4. Calcolo/Applicazione della Normalizzazione (Risoluzione Data Leakage)
        if split == 'train':
            # Calcola le statistiche SOLO sui dati di training (correnti)
            self.data_mean = self.data.mean().detach().clone().contiguous()
            self.data_std = self.data.std().detach().clone().contiguous()
            # Gestione del caso dev_std = 0 (per stabilità)
            self.data_std[self.data_std == 0] = 1.0 #! Evita divisione per zero in __getitem__
            print(f"Calculated new mean/std for training split.")
        elif data_mean is not None and data_std is not None:
            # Usa le statistiche fornite (quelle del training)
            self.data_mean = data_mean.cpu().detach().clone().contiguous()
            self.data_std = data_std.cpu().detach().clone().contiguous()
            print(f"Using provided mean/std for {split} split.")
        else:
            raise ValueError(f"For split '{split}', data_mean and data_std must be provided to avoid data leakage.")
        
        # 5. Subject, Task, Run Encoding (Riassegnazione delle etichette locali 0..N-1)
        original_subjects = self.subjects
        unique_subjects_sorted = np.sort(np.unique(original_subjects))
        subject_map = {id: new_id for new_id, id in enumerate(unique_subjects_sorted)}
        self.subjects = np.array([subject_map[id] for id in original_subjects], dtype=original_subjects.dtype)
        
        original_tasks = self.tasks
        unique_tasks_sorted = np.sort(np.unique(original_tasks))
        task_map = {id: new_id for id, new_id in enumerate(unique_tasks_sorted)} 
        self.tasks = np.array([task_map[id] for id in original_tasks], dtype=original_tasks.dtype)
        
        original_runs = self.runs
        unique_runs_sorted = np.sort(np.unique(original_runs))
        run_map = {id: new_id for new_id, id in enumerate(unique_runs_sorted)}
        self.runs = np.array([run_map[id] for id in original_runs], dtype=original_runs.dtype)

        # 6. INDEX MAPPING (Per proprietà e campionamento)
        self.unique_subjects = np.unique(self.subjects).tolist()
        self.unique_tasks = np.unique(self.tasks).tolist()
        self.unique_runs = np.unique(self.runs).tolist()

        self.subject_indices = {s: [] for s in self.unique_subjects}
        self.task_indices = {t: [] for t in self.unique_tasks}
        self.run_indices = {r: [] for r in self.unique_runs}

        self.full_indices = defaultdict(lambda: defaultdict(list))
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.run_indices[r].append(i)
            self.full_indices[s][t].append(i)
        
        # transform everything into torch tensors for indexing
        self.data = self.data.contiguous().detach().clone().float().contiguous()
        self.subjects = torch.as_tensor(self.subjects, dtype=torch.long)
        self.tasks = torch.as_tensor(self.tasks, dtype=torch.long)
        self.runs = torch.as_tensor(self.runs, dtype=torch.long)

        self.reset_sample_counts()
    
    # --- Metodi Essenziali di PyTorch Dataset ---
    
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        
        # Normalizzazione: Applica mean/std calcolati sul training
        # normalized_data = (sample_data - self.data_mean) / (self.data_std + 1e-6) 
        normalized_data = (sample_data ) # TODO: Change this


        
        return normalized_data, self.subjects[idx], self.tasks[idx], self.runs[idx], idx

    # --- Metodi Aggiuntivi per la DelegatedLoader (Funzionalità Invariata) ---

    def reset_sample_counts(self):
        self.total_samples = 0
    
    def get_dataloader(self, num_total_samples=None, batch_size=None, property=None, random_sample=True):
        print("Getting MI dataloader...")
        # L'assunzione è che DelegatedLoader esista e funzioni come un wrapper del Dataset
        delegated_loader = DelegatedLoader(self, property=property, batch_size=batch_size if random_sample else None, length=num_total_samples)
        if not random_sample and batch_size is not None:
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=True, num_workers=0)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=True, num_workers=0)
    
    def sample_by_condition(self, subjects, tasks):
        samples = []
        for s, t in zip(subjects, tasks):
            i = np.random.choice(self.full_indices[s][t])
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        # NOTA: Qui si restituisce il dato NON normalizzato (coerente con l'originale, ma da verificare se voluto)
        return self.data[samples]    
    
    def sample_by_property(self, property):
        
        property = property.lower()
        if property.startswith("s"):
            property_indices = self.subject_indices
        elif property.startswith("t"):
            property_indices = self.task_indices
        elif property.startswith("r"):
            property_indices = self.run_indices
        else:
            raise ValueError("Invalid property")
        
        samples = []
        for indices in property_indices.values():
            i = np.random.choice(indices)
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
        # NOTA: Anche qui si restituisce il dato NON normalizzato
        return samples, self.data[samples], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        samples_torch = torch.as_tensor(samples, dtype=torch.long)
        self.total_samples += batch_size
        # NOTA: Anche qui si restituisce il dato NON normalizzato
        return samples, self.data[samples_torch], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
            # NOTA: Anche qui si restituisce il dato NON normalizzato
            yield i, self.data[i], self.subjects[i], self.tasks[i], self.runs[i]
    
    def batch_iterator(self, batch_size, length):
        num_samples = 0
        while True:
            if length is not None and num_samples + batch_size >= length:
                break
            yield self.sample_batch(batch_size)
            num_samples += batch_size
    
    def property_iterator(self, property, length):
        num_samples = 0
        num_per = 0
        while True:
            if length is not None and num_samples + num_per >= length:
                break
            yield self.sample_by_property(property)
            if length is not None:
                if num_per == 0:
                    property = property.lower()
                    if property.startswith("s"):
                        num_per = len(self.subject_indices)
                    elif property.startswith("t"):
                        num_per = len(self.task_indices)
                    elif property.startswith("r"):
                        num_per = len(self.run_indices)
                    else:
                        raise ValueError("Invalid property")
                num_samples += num_per