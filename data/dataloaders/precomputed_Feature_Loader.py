import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader, IterableDataset
from collections import defaultdict
from data.dataloaders.delegated_loader import DelegatedLoader


class PrecomputedFeatureLoader():
    """Generic loader for pre-computed features from a .pt file."""
    
    def __init__(self, data_dict, split_subjects, split='train', location='cuda'):
        self.data = data_dict["data"]
        
        device_type = self.data.device.type
        self.split = split

        def to_numpy(tensor):
            return tensor.cpu().numpy() if device_type == 'cuda' else tensor.numpy()
        
        # --- 1. Load Data ---
        # print keys of data_dict # Assuming you uncommented this for debugging.
        # print(data_dict.keys()) 
        self.subjects = to_numpy(data_dict["subjects"])
        if "tasks" in data_dict:
            self.tasks = to_numpy(data_dict["tasks"])
        else:
            self.tasks = np.ones_like(self.subjects)    
        
        if "runs" in data_dict:
            self.runs = to_numpy(data_dict["runs"])
        else:
            self.runs = np.ones_like(self.subjects)

        if 'data_mean' in data_dict:
            #self.data_mean = 1
            self.data_mean = data_dict['data_mean'].detach().clone().contiguous().to(location)
        else:
            self.data_mean = self.data.mean(dim=0).to(location)
        
        if 'data_std' in data_dict:
            #self.data_std = 0
            self.data_std = data_dict['data_std'].detach().clone().contiguous().to(location)
        else:
            self.data_std = self.data.std(dim=0).to(location)
            
        # These are used to filter the incoming data:
        self.unique_subjects_filter = split_subjects 
        
        # --- 2. Filter Data Indices based on split_subjects ---
        # Note: We keep the original labels for filtering before remapping.
        data_indices = []
        for i, s in enumerate(self.subjects):
            if s in self.unique_subjects_filter:
                data_indices.append(i)
        
        # Apply filtering
        if location == "cpu":
            self.data = self.data[data_indices].float().contiguous().detach().cpu()
        elif location == "cuda":
            self.data = self.data[data_indices].float().contiguous().detach().cuda()
        
        torch.cuda.empty_cache()
        
        self.subjects = np.ascontiguousarray(self.subjects[data_indices])
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        # --- 3. CRITICAL: Contiguous Label Encoding (Zero-Indexing) ---
        
        # Subject Encoding (Ensures [2, 4, 5] maps to [0, 1, 2])
        original_subjects = self.subjects
        unique_subjects_sorted = np.sort(np.unique(original_subjects))
        subject_map = {id: new_id for new_id, id in enumerate(unique_subjects_sorted)}
        self.subjects = np.array([subject_map[id] for id in original_subjects], dtype=original_subjects.dtype)
        
        # Task Encoding (Ensures task labels like [1, 2, 5] map to [0, 1, 2])
        original_tasks = self.tasks
        unique_tasks_sorted = np.sort(np.unique(original_tasks))
        task_map = {id: new_id for new_id, id in enumerate(unique_tasks_sorted)}
        self.tasks = np.array([task_map[id] for id in original_tasks], dtype=original_tasks.dtype)
        
        # Runs Encoding (Ensures run labels are also contiguous 0, 1, 2...)
        original_runs = self.runs
        unique_runs_sorted = np.sort(np.unique(original_runs))
        run_map = {id: new_id for new_id, id in enumerate(unique_runs_sorted)}
        self.runs = np.array([run_map[id] for id in original_runs], dtype=original_runs.dtype)
        
        # --- 4. Final Unique Lists and Indices Setup ---
        
        # These unique lists now contain the contiguous, zero-indexed labels [0, 1, 2, ...]
        self.unique_subjects = np.unique(self.subjects).tolist()
        self.unique_tasks = np.unique(self.tasks).tolist()
        self.unique_runs = np.unique(self.runs).tolist()

        # Initialize tracking dictionaries using the NEW zero-indexed labels
        self.subject_indices = {s: [] for s in self.unique_subjects}
        self.task_indices = {t: [] for t in self.unique_tasks}
        self.run_indices = {r: [] for r in self.unique_runs}
        
        self.total_samples = 0 # Initialization for sample tracking
        
        self.full_indices = defaultdict(lambda: defaultdict(list))
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            self.subject_indices[s].append(i)
            self.task_indices[t].append(i)
            self.run_indices[r].append(i)
            self.full_indices[s][t].append(i)

    def reset_sample_counts(self):
        self.total_samples = 0
    
    def get_dataloader(self, num_total_samples=None, batch_size=None, property=None, random_sample=True):
        delegated_loader = DelegatedLoader(self, property=property, batch_size=batch_size if random_sample else None, length=num_total_samples)
        if not random_sample and batch_size is not None:
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=True, num_workers=16)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=True, num_workers=16)
    
    def sample_by_condition(self, subjects, tasks):
        samples = []
        for s, t in zip(subjects, tasks):
            i = np.random.choice(self.full_indices[s][t])
            samples.append(i)
        samples = np.array(samples)
        self.total_samples += len(samples)
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
        return samples, self.data[samples], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def sample_batch(self, batch_size):
        samples = np.random.randint(0, self.size, size=batch_size)
        self.total_samples += batch_size
        return samples, self.data[samples], self.subjects[samples], self.tasks[samples], self.runs[samples]
    
    def iterator(self):
        for i in range(self.size):
            self.total_samples += 1
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

