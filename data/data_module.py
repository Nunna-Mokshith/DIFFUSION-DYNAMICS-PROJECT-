"""
Professional Data Module for Multi-Dataset Support
=================================================
Supports QM9, ZINC, and MoleculeNet.
"""

import os
import torch
from torch_geometric.datasets import QM9, ZINC, MoleculeNet
from torch_geometric.loader import DataLoader

class DataModule:
    def __init__(self, dataset_name="qm9", root="./data", batch_size=32, max_samples=None):
        self.dataset_name = dataset_name.lower()
        self.root = root
        self.batch_size = batch_size
        self.max_samples = max_samples

    def get_loaders(self):
        if self.dataset_name == "qm9":
            dataset = QM9(root=self.root)
            # Standard property normalization for QM9
            # Property 10 is Gibbs Free Energy
            pass 
        elif self.dataset_name == "zinc":
            dataset = ZINC(root=self.root, subset=True, split="train")
        elif self.dataset_name == "moleculenet":
            dataset = MoleculeNet(root=self.root, name="ESOL")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if self.max_samples:
            indices = torch.randperm(len(dataset))[:self.max_samples]
            dataset = dataset[indices]

        split = int(0.9 * len(dataset))
        train_d = dataset[:split]
        val_d   = dataset[split:]

        train_loader = DataLoader(train_d, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_d,   batch_size=self.batch_size, shuffle=False)
        
        return train_loader, val_loader

    @staticmethod
    def get_atom_mapping(dataset_name):
        if dataset_name == "qm9":
            return {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
        elif dataset_name == "zinc":
            # ZINC has different atom types
            return {0: 6, 1: 7, 2: 8, 3: 16, 4: 17, 5: 9, 6: 15, 7: 53} # approx map
        return {0: 6}
