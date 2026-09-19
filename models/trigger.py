import torch
import torch.nn as nn
import numpy as np

class TriggerModule(nn.Module):
    def __init__(self, image_shape=(3, 32, 32)):
        super().__init__()
        c, h, w = image_shape

        init_mask = np.random.normal(0, 1, (1, h, w)).astype(np.float32)  # shape: (1, H, W)
        init_pattern = np.random.normal(0, 1, (c, h, w)).astype(np.float32)  # shape: (C, H, W)

        # Both tensors are optimized jointly with the attack objective.
        self.mask = nn.Parameter(torch.from_numpy(init_mask).clamp(0, 1))
        self.pattern = nn.Parameter(torch.from_numpy(init_pattern).clamp(0, 1))

    def forward(self, dummy_input=None):
        return self.pattern, self.mask
