import numpy as np
import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from data.dataloaders.delegated_loader import DelegatedLoader

class CustomLoaderSleep():
    
    def __init__(self, data_dict, split_subjects, split='train', location = 'cuda', last_sample=None):
        self.data = data_dict["data"]
        
        device_type = self.data.device.type
        self.split = split

        def to_numpy(tensor):
            return tensor.cpu().numpy() if device_type == 'cuda' else tensor.numpy()

        self.subjects = to_numpy(data_dict["subjects"])
        self.tasks = to_numpy(data_dict["tasks"])
        self.runs = to_numpy(data_dict["runs"])


        self.data_mean = data_dict['data_mean'].detach().clone().contiguous().cuda()
        self.data_std = data_dict['data_std'].detach().clone().contiguous().cuda()
        
        #self.data_mean = self.data.mean().detach().clone().contiguous().cuda()
        #self.data_std = self.data.std().detach().clone().contiguous().cuda()
        
        # Sleep task definitions (typically 1-indexed)
        self.task_to_label = {
            1: "Sleep stage W", 2: "Sleep stage 1", 3: "Sleep stage 2", 
            4: "Sleep stage 3/4", 5: "Sleep stage R"
        }
        self.run_labels = ['First night', 'Second night']

        # --- Filter definitions (using original IDs) ---
        self.unique_subjects_filter = split_subjects
        self.unique_tasks_filter = list(self.task_to_label.keys()) # [1, 2, 3, 4, 5]
        self.unique_runs_filter = list(range(1, len(self.run_labels) + 1)) # [1, 2]

        # --- Filter data down to the current split's samples ---
        data_indices = []
        for i, (s, t, r) in enumerate(zip(self.subjects, self.tasks, self.runs)):
            if s not in self.unique_subjects_filter:
                continue
            if t not in self.unique_tasks_filter:
                continue
            if r not in self.unique_runs_filter:
                continue
            data_indices.append(i)
            
        if location == "cpu":
            self.data = self.data[data_indices].float().contiguous().detach().cpu()
        elif location == "cuda":
            self.data = self.data[data_indices].float().contiguous().detach().cuda() 
        torch.cuda.empty_cache()
        
        self.data = (self.data - self.data_mean) / (self.data_std + 1e-6)
        
        self.subjects = np.ascontiguousarray(self.subjects[data_indices]) 
        self.tasks = np.ascontiguousarray(self.tasks[data_indices])
        self.runs = np.ascontiguousarray(self.runs[data_indices])
        self.size = len(self.data)

        # --- CRITICAL FIX: Contiguous Label Encoding ---
        
        # 1. Subject Encoding 
        original_subjects = self.subjects
        unique_subjects_sorted = np.sort(np.unique(original_subjects))
        subject_map = {id: new_id for new_id, id in enumerate(unique_subjects_sorted)}
        self.subjects = np.array([subject_map[id] for id in original_subjects], dtype=original_subjects.dtype)
        
        # 2. Task Encoding (Maps original task IDs [1, 2, 3, 4, 5] to [0, 1, 2, 3, 4])
        original_tasks = self.tasks
        unique_tasks_sorted = np.sort(np.unique(original_tasks))
        task_map = {id: new_id for new_id, id in enumerate(unique_tasks_sorted)}
        self.tasks = np.array([task_map[id] for id in original_tasks], dtype=original_tasks.dtype)
        
        # 3. Runs Encoding (Maps original run IDs [1, 2] to [0, 1])
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
            return DataLoader(delegated_loader, batch_size=batch_size, pin_memory=False)
        return DataLoader(delegated_loader, batch_size=None, pin_memory=False)
    
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
        # Return batches of size batch_size, until length is reached.
        num_samples = 0
        while True:
            if length is not None and num_samples + batch_size >= length:
                break
            yield self.sample_batch(batch_size)
            num_samples += batch_size
    
    def property_iterator(self, property, length):
        # Return batches with unique property (subject/task/run), until length is reached.
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

