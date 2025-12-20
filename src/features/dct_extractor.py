import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.fftpack import dct


class DCTFeatureExtractor(nn.Module):
    def __init__(self, block_size: int = 8):
        super().__init__()
        self.block_size = block_size
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        bs = self.block_size
        device = images.device
        dtype = images.dtype
        
        H_pad = (bs - H % bs) % bs
        W_pad = (bs - W % bs) % bs
        if H_pad > 0 or W_pad > 0:
            images = F.pad(images, (0, W_pad, 0, H_pad), mode='reflect')
        
        B, C, H_new, W_new = images.shape
        blocks = images.unfold(2, bs, bs).unfold(3, bs, bs)
        blocks = blocks.contiguous().view(B, C, -1, bs, bs)
        
        dct_features_list = []
        for b in range(B):
            batch_dct = []
            for c in range(C):
                channel_blocks = blocks[b, c].cpu().numpy()
                dct_result = dct(dct(channel_blocks, axis=-1, norm='ortho'), axis=-2, norm='ortho')
                batch_dct.append(dct_result)
            dct_features_list.append(np.stack(batch_dct, axis=0))
        
        dct_features = torch.from_numpy(np.stack(dct_features_list, axis=0))
        dct_features = dct_features.to(device=device, dtype=dtype)
        dct_flat = dct_features.view(B, -1)
        return dct_flat
