import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict


class SRMExtractor(nn.Module):
    def __init__(self, freeze_weights: bool = True):
        super().__init__()
        kernel_types = ['horizontal', 'vertical', 'diagonal_1', 'diagonal_2', 'edge_3x3']
        self.filters = nn.ModuleList([self._create_filter(self._get_kernel(kt)) for kt in kernel_types])
        self.num_filters = len(self.filters)
        
        if freeze_weights:
            for filt in self.filters:
                filt.weight.requires_grad = False
                if filt.bias is not None:
                    filt.bias.requires_grad = False
    
    def _get_kernel(self, kernel_type: str) -> np.ndarray:
        kernels = {
            'horizontal': np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]),
            'vertical': np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]),
            'diagonal_1': np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]),
            'diagonal_2': np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]),
            'edge_3x3': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        }
        return kernels.get(kernel_type, np.zeros((3, 3)))
    
    def _create_filter(self, kernel: np.ndarray) -> nn.Conv2d:
        conv = nn.Conv2d(1, 1, kernel_size=kernel.shape[0], padding=kernel.shape[0]//2, bias=False)
        kernel_tensor = torch.from_numpy(kernel).float().unsqueeze(0).unsqueeze(0)
        conv.weight.data = kernel_tensor
        return conv
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        gray = 0.299 * images[:, 0:1] + 0.587 * images[:, 1:2] + 0.114 * images[:, 2:3]
        
        srm_maps = []
        for filt in self.filters:
            filtered = filt(gray)
            srm_maps.append(filtered)
        
        srm_output = torch.cat(srm_maps, dim=1)
        return srm_output
