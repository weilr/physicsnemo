# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
from mamba_ssm import Mamba
from torch.cuda.amp import autocast

class MambaLayer(nn.Module):
    """Mamba层，用于处理网格特征。
    
    将输入特征展平为序列形式，通过Mamba处理后再恢复原始形状。
    支持自动类型转换（float16 -> float32）以提高数值稳定性。
    """
    
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        """
        Args:
            dim: 输入特征维度
            d_state: SSM状态扩展因子
            d_conv: 局部卷积宽度
            expand: 块扩展因子
        """
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
        )
    
    @autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征，形状为[B, C, ...]
            
        Returns:
            处理后的特征，与输入形状相同
        """
        # 记录原始数据类型
        orig_dtype = x.dtype
        
        # 自动转换float16为float32以提高数值稳定性
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
            
        B, C = x.shape[:2]
        assert C == self.dim, f"Channel dimension mismatch: {C} != {self.dim}"
        
        # 记录原始维度并展平为序列
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  # [B, N, C]
        
        # 应用层归一化
        x_norm = self.norm(x_flat)
        
        # Mamba处理
        x_mamba = self.mamba(x_norm)
        
        # 恢复原始形状
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        # 恢复原始数据类型
        if orig_dtype != out.dtype:
            out = out.type(orig_dtype)
            
        return out