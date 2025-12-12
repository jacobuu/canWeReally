import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from data.dataloaders.delegated_loader import DelegatedLoader



class CustomLoaderERP():
    '''Custom data loader for ERP data with filtering and indexing capabilities.'''
    
    def __init__(self, data_dict, split='train'):
        self.split = split
        self.data = data_dict["features"] if "features" in data_dict else data_dict["data"]
        self.size = len(self.data)
        self.subjects = data_dict["subjects"].numpy()
        self.tasks = data_dict["tasks"].numpy()
        # Ensure runs are >= 1 before converting to numpy
        self.runs = data_dict["runs"].clamp(min=1).numpy()
        
        self.data_mean = data_dict['data_mean'].detach().clone().contiguous().cuda()
        self.data_std = data_dict['data_std'].detach().clone().contiguous().cuda()
        
        # --- Pre-defined splits (using original subject IDs) ---
        dev_splits = [4, 7, 27, 33]
        test_splits = [5, 14, 15, 20, 22, 23, 26, 29]
        train_splits = [1, 2, 3, 6, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 24, 25, 28, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40]
        
        if split == 'dev':
            self.unique_subjects_filter = dev_splits
        elif split == 'test':
            self.unique_subjects_filter = test_splits
        elif split == 'train':
            self.unique_subjects_filter = train_splits
        elif split == 'N170':
            self.unique_subjects_filter = list(range(1, 41))
        else:
            raise ValueError('Invalid split')
        
        self.task_to_label = data_dict['labels']
        self.run_labels = ['ERN+LRP', 'MMN', 'N2pc', 'N170', 'N400', 'P3']
        
        # --- Filter unique tasks/runs based on the current split/paradigm ---
        if split != 'N170':
            unique_tasks_filter = [t for t, l in self.task_to_label.items() if not l.startswith('N170')]
            unique_runs_filter = [r + 1 for r, l in enumerate(self.run_labels) if l != 'N170']
        else:
            unique_tasks_filter = [t for t, l in self.task_to_label.items() if l.startswith('N170')]
            unique_runs_filter = [r + 1 for r, l in enumerate(self.run_labels) if l == 'N170']

        # --- Filter data down to the current split's samples ---
        data_indices = []
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            if s not in self.unique_subjects_filter: # Filter by subject split
                continue
            if t not in unique_tasks_filter:         # Filter by task type
                continue
            if r not in unique_runs_filter:          # Filter by run type
                continue
            data_indices.append(i)
            
        self.data = self.data[data_indices].float().contiguous().detach().clone()
        self.subjects = np.ascontiguousarray(self.subjects[data_indices])
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        # --- CRITICAL FIX: Contiguous Label Encoding (Map non-contiguous IDs to 0, 1, 2...) ---
        
        # 1. Subject Encoding (e.g., [5, 14, 15] -> [0, 1, 2])
        original_subjects = self.subjects
        unique_subjects_sorted = np.sort(np.unique(original_subjects))
        subject_map = {id: new_id for new_id, id in enumerate(unique_subjects_sorted)}
        self.subjects = np.array([subject_map[id] for id in original_subjects], dtype=original_subjects.dtype)
        
        # 2. Task Encoding (e.g., [1, 2, 5, 6] -> [0, 1, 2, 3])
        original_tasks = self.tasks
        unique_tasks_sorted = np.sort(np.unique(original_tasks))
        task_map = {id: new_id for new_id, id in enumerate(unique_tasks_sorted)}
        self.tasks = np.array([task_map[id] for id in original_tasks], dtype=original_tasks.dtype)
        
        # 3. Runs Encoding (Ensure run IDs are also 0-indexed and contiguous)
        original_runs = self.runs
        unique_runs_sorted = np.sort(np.unique(original_runs))
        run_map = {id: new_id for new_id, id in enumerate(unique_runs_sorted)}
        self.runs = np.array([run_map[id] for id in original_runs], dtype=original_runs.dtype)

        # --- Final unique lists and index maps use the new contiguous labels ---
        self.unique_subjects = np.unique(self.subjects).tolist()
        self.unique_tasks = np.unique(self.tasks).tolist()
        self.unique_runs = np.unique(self.runs).tolist()

        self.subject_indices = {s: [] for s in self.unique_subjects}
        self.task_indices = {t: [] for t in self.unique_tasks}
        self.run_indices = {r: [] for r in self.unique_runs}
        
        self.total_samples = 0 # Ensure initialization
        
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
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=True)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=True)
    
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
        for indices in property_indices.values(): # {1:[(1,1,1), (1,2,1)], 2:[(2,1,1), (2,2,1)]} [(1,1,1), (1,2,1)], [(2,1,1), (2,2,1)]
            i = np.random.choice(indices)           # (1,1,1), (2,1,1)
            samples.append(i)                       # [(1,1,1), (2,1,1)]
        samples = np.array(samples)                 # [(1,1,1), (2,1,1)]
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
