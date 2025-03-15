import torch
import numpy as np
from torch.utils.data import Sampler, DataLoader

class BalancedTilehSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.labels = np.array(dataset.tiles_df.merge(dataset.slides_df, on="slide_num")['tag'])  # Extract labels
        self.label_to_indices = self._group_by_label()
        self.min_class_size = min(len(indices) for indices in self.label_to_indices.values())
        
        # Ensure batch size is even and balanced
        assert batch_size % len(self.label_to_indices) == 0, "Batch size must be divisible by the number of classes"
        
    def _group_by_label(self):
        """Create a dictionary mapping labels to index lists."""
        label_to_indices = {}
        for idx, label in enumerate(self.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __iter__(self):
        indices = []
        num_samples_per_class = self.batch_size // len(self.label_to_indices)
        
        # Ensure we have enough samples
        for _ in range(self.min_class_size // num_samples_per_class):
            batch = []
            for label, idx_list in self.label_to_indices.items():
                sampled_indices = np.random.choice(idx_list, num_samples_per_class, replace=False)
                batch.extend(sampled_indices)
            np.random.shuffle(batch)
            indices.append(batch)

        return iter(indices)

    def __len__(self):
        return self.min_class_size // (self.batch_size // len(self.label_to_indices))
