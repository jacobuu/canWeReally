import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, IterableDataset


class DelegatedLoader(IterableDataset):
    '''
    Wrapper to delegate loading to custom loader with optional batching/property filtering.'''
    
    def __init__(self, loader, property=None, batch_size=None, length=None):
        self.loader = loader
        self._property = property
        self._batch_size = batch_size
        self._length = length
    
    def __len__(self):
        if self._batch_size is not None or self._property is not None:
            if self._length is not None:
                if self._batch_size is not None:
                    return self._length // self._batch_size
                return self._length
            return None
        return self.size
    
    def __iter__(self):
        if self._batch_size is not None or self._property is not None:
            if self._batch_size is not None:
                return self.loader.batch_iterator(self._batch_size, self._length)
            elif self._property is not None:
                return self.loader.property_iterator(self._property, self._length)
        else:
            return self.loader.iterator()
