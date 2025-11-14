from abc import abstractmethod
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Union

# 模态类型定义
MODALITY_TYPES = ['image', 'property', 'structure']


class ModalityEmbedding(nn.Module):
    """多模态嵌入模块，为不同类型输入生成模态特定嵌入"""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.modal_embeddings = nn.Embedding(len(MODALITY_TYPES), embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, modal_indices: th.Tensor) -> th.Tensor:
        emb = self.modal_embeddings(modal_indices)
        return self.proj(emb)


class CrossModalityAttention(nn.Module):
    """跨模态注意力模块，实现不同模态特征的交互"""

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.attention_scale = self.head_dim ** -0.5

    def forward(self, x: th.Tensor, context: th.Tensor) -> th.Tensor:
        """
        x: 目标模态特征 [B, T, C]
        context: 上下文模态特征 [B, S, C]
        """
        B, T, C = x.shape
        _, S, _ = context.shape

        # 生成查询、键、值
        qkv = self.qkv_proj(self.norm(x)).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # [B, H, T, D], [B, H, T, D], [B, H, T, D]

        # 上下文特征作为额外键值对
        context_kv = self.qkv_proj(self.norm(context)).reshape(B, S, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        context_k, context_v = context_kv.unbind(0)  # [B, H, S, D], [B, H, S, D]

        # 合并键值对
        k = th.cat([k, context_k], dim=2)  # [B, H, T+S, D]
        v = th.cat([v, context_v], dim=2)  # [B, H, T+S, D]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.attention_scale
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)

        return self.out_proj(out) + x


class PropertyAwareBlock(nn.Module):
    """属性感知模块，将属性特征融入3D重建流程"""

    def __init__(self, in_channels: int, prop_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.prop_channels = prop_channels

        # 属性特征处理
        self.prop_proj = nn.Sequential(
            nn.Linear(prop_channels, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )

        # 空间特征处理
        self.spatial_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)

        # 融合模块
        self.fusion = nn.Sequential(
            nn.LayerNorm(out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=1)
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: th.Tensor, prop: th.Tensor) -> th.Tensor:
        """
        x: 3D空间特征 [B, C, D, H, W]
        prop: 属性特征 [B, P]
        """
        B, C, D, H, W = x.shape

        # 处理空间特征
        spatial_feat = self.spatial_conv(x)

        # 处理属性特征并广播到空间维度
        prop_feat = self.prop_proj(prop)  # [B, O]
        prop_feat = prop_feat.view(B, -1, 1, 1, 1).expand(-1, -1, D, H, W)  # [B, O, D, H, W]

        # 门控融合
        gate = self.gate(th.cat([spatial_feat, prop_feat], dim=1))
        fused = gate * spatial_feat + (1 - gate) * prop_feat

        return self.fusion(fused)


class TimestepBlock(nn.Module):
    """时间步嵌入感知模块"""

    @abstractmethod
    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        pass


class ResidualBlock(TimestepBlock):
    """增强版残差块，支持3D数据和多模态特征融合"""

    def __init__(
            self,
            channels: int,
            emb_channels: int,
            out_channels: Optional[int] = None,
            dims: int = 3,
            dropout: float = 0.0,
            use_scale_shift_norm: bool = True,
            use_checkpoint: bool = False,
            has_attention: bool = False,
            num_heads: int = 4
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.use_checkpoint = use_checkpoint

        # 主卷积序列
        self.in_layers = nn.Sequential(
            self._norm(channels, dims),
            nn.SiLU(),
            self._conv(dims, channels, self.out_channels, 3, padding=1)
        )

        # 时间嵌入处理
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        # 输出卷积序列
        self.out_layers = nn.Sequential(
            self._norm(self.out_channels, dims),
            nn.SiLU(),
            nn.Dropout(dropout),
            self._zero_conv(dims, self.out_channels, self.out_channels, 3, padding=1)
        )

        # 跳跃连接
        if self.out_channels != channels:
            self.skip_conv = self._conv(dims, channels, self.out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        # 注意力模块
        self.attention = None
        if has_attention:
            self.attention = self._create_attention(self.out_channels, num_heads, dims)

        self.use_scale_shift_norm = use_scale_shift_norm

    def _norm(self, channels: int, dims: int) -> nn.Module:
        if dims == 3:
            return nn.GroupNorm(32, channels)
        return nn.GroupNorm(32, channels)

    def _conv(self, dims: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        if dims == 1:
            return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)
        elif dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)

    def _zero_conv(self, dims: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        conv = self._conv(dims, in_channels, out_channels, kernel_size, **kwargs)
        nn.init.zeros_(conv.weight)
        return conv

    def _create_attention(self, channels: int, num_heads: int, dims: int) -> nn.Module:
        if dims == 3:
            return Attention3D(channels, num_heads)
        elif dims == 2:
            return Attention2D(channels, num_heads)
        else:
            return Attention1D(channels, num_heads)

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        return self._forward(x, emb)

    def _forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        h = self.in_layers(x)

        # 处理时间嵌入
        emb_out = self.emb_proj(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out.unsqueeze(-1)

        # 应用尺度和偏移
        if self.use_scale_shift_norm:
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        else:
            h = h + emb_out

        # 应用注意力
        if self.attention is not None:
            h = self.attention(h)

        # 输出处理和残差连接
        h = self.out_layers(h)
        return h + self.skip_conv(x)


class Attention3D(nn.Module):
    """3D注意力模块，支持体积数据的自注意力"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv3d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, D, H, W = x.shape
        spatial_size = D * H * W

        # 归一化和投影
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(B, 3, self.num_heads, self.head_dim, spatial_size)
        q, k, v = qkv.unbind(1)  # [B, H, D, N]

        # 注意力计算
        attn = (q.transpose(-2, -1) @ k) * self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)

        # 输出投影
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # [B, H, D, N]
        out = out.contiguous().view(B, C, D, H, W)

        return self.out(out) + x


class Attention2D(nn.Module):
    """2D注意力模块，用于处理图像模态"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, H, W = x.shape
        spatial_size = H * W

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(B, 3, self.num_heads, self.head_dim, spatial_size)
        q, k, v = qkv.unbind(1)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, H, W)

        return self.out(out) + x


class Attention1D(nn.Module):
    """1D注意力模块，用于处理序列属性数据"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(32, channels)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: th.Tensor) -> th.Tensor:
        B, C, L = x.shape

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).view(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)
        out = out.contiguous().view(B, C, L)

        return self.out(out) + x


class Upsample(nn.Module):
    """多维度上采样模块，支持3D/2D/1D数据"""

    def __init__(self, channels: int, dims: int = 3, use_conv: bool = True):
        super().__init__()
        self.dims = dims
        self.use_conv = use_conv
        if use_conv:
            self.conv = self._create_conv(channels, channels, 3, padding=1)

    def _create_conv(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        if self.dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        elif self.dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.dims == 3:
            x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=False)
        elif self.dims == 2:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode="linear", align_corners=False)

        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """多维度下采样模块"""

    def __init__(self, channels: int, dims: int = 3, use_conv: bool = True):
        super().__init__()
        self.dims = dims
        self.use_conv = use_conv
        if use_conv:
            stride = (1, 2, 2) if dims == 3 else 2
            self.conv = self._create_conv(channels, channels, 3, stride=stride, padding=1)
        else:
            self.pool = self._create_pool()

    def _create_conv(self, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        if self.dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        elif self.dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def _create_pool(self) -> nn.Module:
        if self.dims == 3:
            return nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        elif self.dims == 2:
            return nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            return nn.AvgPool1d(kernel_size=2, stride=2)

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.use_conv:
            return self.conv(x)
        else:
            return self.pool(x)


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """时间步嵌入序列模块，支持带时间嵌入的层序列"""

    def forward(self, x: th.Tensor, emb: th.Tensor) -> th.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class MultiModalEncoder(nn.Module):
    """多模态编码器，处理不同类型的输入模态"""

    def __init__(
            self,
            modal_specs: Dict[str, int],
            hidden_dim: int = 64,
            out_dim: int = 128,
            dims: int = 3
    ):
        super().__init__()
        self.modal_specs = modal_specs
        self.modal_encoders = nn.ModuleDict()
        self.modal_embedding = ModalityEmbedding(hidden_dim)

        # 为每种模态创建编码器
        for modal, in_channels in modal_specs.items():
            if modal == 'image':
                self.modal_encoders[modal] = self._build_image_encoder(in_channels, hidden_dim, out_dim, dims)
            elif modal == 'property':
                self.modal_encoders[modal] = self._build_property_encoder(in_channels, hidden_dim, out_dim)
            elif modal == 'structure':
                self.modal_encoders[modal] = self._build_structure_encoder(in_channels, hidden_dim, out_dim, dims)

        # 跨模态融合
        self.cross_attention = CrossModalityAttention(out_dim)
        self.fusion_proj = nn.Linear(out_dim, out_dim)

    def _build_image_encoder(self, in_channels: int, hidden_dim: int, out_dim: int, dims: int) -> nn.Module:
        return nn.Sequential(
            self._conv(dims, in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim, hidden_dim * 4, hidden_dim * 2, dims=dims, has_attention=True),
            Downsample(hidden_dim * 2, dims=dims),
            ResidualBlock(hidden_dim * 2, hidden_dim * 4, out_dim, dims=dims),
        )

    def _build_property_encoder(self, in_channels: int, hidden_dim: int, out_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, out_dim),
        )

    def _build_structure_encoder(self, in_channels: int, hidden_dim: int, out_dim: int, dims: int) -> nn.Module:
        return nn.Sequential(
            self._conv(dims, in_channels, hidden_dim, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(hidden_dim, hidden_dim * 4, hidden_dim * 2, dims=dims),
            Downsample(hidden_dim * 2, dims=dims),
            ResidualBlock(hidden_dim * 2, hidden_dim * 4, out_dim, dims=dims, has_attention=True),
        )

    def _conv(self, dims: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        if dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        elif dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def forward(self, inputs: Dict[str, th.Tensor]) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        # 编码每种模态
        encoded = {}
        modal_indices = []
        for i, (modal, x) in enumerate(inputs.items()):
            if modal not in self.modal_encoders:
                continue
            # 添加模态嵌入
            modal_emb = self.modal_embedding(th.tensor(i, device=x.device).repeat(x.shape[0]))
            if modal == 'property':
                # 属性特征处理
                feat = self.modal_encoders[modal](x)
                feat = feat + modal_emb
            else:
                # 空间特征处理
                feat = self.modal_encoders[modal](x)
                # 全局池化获取特征向量
                if feat.dim() == 5:  # 3D
                    feat = F.adaptive_avg_pool3d(feat, 1).view(x.shape[0], -1)
                elif feat.dim() == 4:  # 2D
                    feat = F.adaptive_avg_pool2d(feat, 1).view(x.shape[0], -1)
                feat = feat + modal_emb
            encoded[modal] = feat
            modal_indices.append(modal)

        # 跨模态注意力融合
        if len(encoded) == 0:
            raise ValueError("No valid modalities provided")

        # 以结构模态为主要目标
        main_modal = 'structure' if 'structure' in encoded else modal_indices[0]
        main_feat = encoded[main_modal].unsqueeze(1)  # [B, 1, D]

        # 融合其他模态
        context_feats = []
        for modal in modal_indices:
            if modal != main_modal:
                context_feats.append(encoded[modal].unsqueeze(1))  # [B, 1, D]

        if context_feats:
            context = th.cat(context_feats, dim=1)  # [B, S, D]
            fused = self.cross_attention(main_feat, context).squeeze(1)  # [B, D]
        else:
            fused = main_feat.squeeze(1)

        return self.fusion_proj(fused), encoded


class MultiModalDiffusionUNet(nn.Module):
    """多模态属性感知扩散UNet模型，支持3D重建"""

    def __init__(
            self,
            image_size: int,
            in_channels: int,
            model_channels: int = 64,
            out_channels: int = 3,
            num_res_blocks: int = 2,
            attention_resolutions: Tuple[int, ...] = (16, 8),
            dropout: float = 0.1,
            channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
            dims: int = 3,
            num_classes: Optional[int] = None,
            use_checkpoint: bool = False,
            use_fp16: bool = False,
            num_heads: int = 4,
            modal_specs: Dict[str, int] = None,
            prop_channels: int = 32
    ):
        super().__init__()
        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dims = dims
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32

        # 多模态编码器
        self.modal_encoder = MultiModalEncoder(
            modal_specs=modal_specs or {'image': 3, 'property': 10, 'structure': 1},
            hidden_dim=model_channels,
            out_dim=model_channels * 4,
            dims=dims
        )

        # 时间嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )

        # 类别嵌入（如果需要）
        if num_classes is not None:
            self.label_embed = nn.Embedding(num_classes, time_embed_dim)

        # 输入处理
        self.input_proj = self._conv(dims, in_channels, model_channels, 3, padding=1)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        current_channels = model_channels
        down_block_chans = [current_channels]
        ds = 1

        for level, mult in enumerate(channel_mult):
            # 添加残差块
            for _ in range(num_res_blocks):
                has_attention = ds in attention_resolutions
                block = TimestepEmbedSequential(
                    ResidualBlock(
                        current_channels,
                        time_embed_dim,
                        model_channels * mult,
                        dims=dims,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        has_attention=has_attention,
                        num_heads=num_heads
                    )
                )
                self.down_blocks.append(block)
                current_channels = model_channels * mult
                down_block_chans.append(current_channels)

                # 添加属性感知模块（每两个残差块后）
                if _ % 2 == 1:
                    self.down_blocks.append(TimestepEmbedSequential(
                        PropertyAwareBlock(current_channels, prop_channels, current_channels)
                    ))

            # 添加下采样（最后一层除外）
            if level != len(channel_mult) - 1:
                self.down_blocks.append(TimestepEmbedSequential(
                    Downsample(current_channels, dims=dims, use_conv=True)
                ))
                down_block_chans.append(current_channels)
                ds *= 2

        # 中间块
        self.middle_block = TimestepEmbedSequential(
            ResidualBlock(
                current_channels,
                time_embed_dim,
                current_channels,
                dims=dims,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                has_attention=True,
                num_heads=num_heads
            ),
            PropertyAwareBlock(current_channels, prop_channels, current_channels),
            ResidualBlock(
                current_channels,
                time_embed_dim,
                current_channels,
                dims=dims,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
                has_attention=True,
                num_heads=num_heads
            )
        )

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        rev_channel_mult = list(reversed(channel_mult))

        for level, mult in enumerate(rev_channel_mult):
            # 添加残差块
            for i in range(num_res_blocks + 1):
                has_attention = (ds // (2 ** (level + 1))) in attention_resolutions
                block = TimestepEmbedSequential(
                    ResidualBlock(
                        current_channels + down_block_chans.pop(),
                        time_embed_dim,
                        model_channels * mult,
                        dims=dims,
                        dropout=dropout,
                        use_checkpoint=use_checkpoint,
                        has_attention=has_attention,
                        num_heads=num_heads
                    )
                )
                self.up_blocks.append(block)
                current_channels = model_channels * mult

                # 添加属性感知模块
                if i % 2 == 0 and current_channels > model_channels:
                    self.up_blocks.append(TimestepEmbedSequential(
                        PropertyAwareBlock(current_channels, prop_channels, current_channels)
                    ))

            # 添加上采样（最后一层除外）
            if level != len(rev_channel_mult) - 1:
                self.up_blocks.append(TimestepEmbedSequential(
                    Upsample(current_channels, dims=dims, use_conv=True)
                ))
                ds //= 2

        # 输出层
        self.out = nn.Sequential(
            self._norm(current_channels),
            nn.SiLU(),
            self._zero_conv(dims, current_channels, out_channels, 3, padding=1)
        )

    def _conv(self, dims: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        if dims == 3:
            return nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        elif dims == 2:
            return nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        else:
            return nn.Conv1d(in_channels, out_channels, kernel_size, **kwargs)

    def _norm(self, channels: int) -> nn.Module:
        return nn.GroupNorm(32, channels)

    def _zero_conv(self, dims: int, in_channels: int, out_channels: int, kernel_size: int, **kwargs) -> nn.Module:
        conv = self._conv(dims, in_channels, out_channels, kernel_size, **kwargs)
        nn.init.zeros_(conv.weight)
        return conv

    def convert_to_fp16(self):
        """转换为FP16精度"""
        self.modal_encoder = self.modal_encoder.half()
        self.time_embed = self.time_embed.half()
        self.down_blocks = self.down_blocks.half()
        self.middle_block = self.middle_block.half()
        self.up_blocks = self.up_blocks.half()
        self.out = self.out.half()
        if self.num_classes is not None:
            self.label_embed = self.label_embed.half()

    def convert_to_fp32(self):
        """转换为FP32精度"""
        self.modal_encoder = self.modal_encoder.float()
        self.time_embed = self.time_embed.float()
        self.down_blocks = self.down_blocks.float()
        self.middle_block = self.middle_block.float()
        self.up_blocks = self.up_blocks.float()
        self.out = self.out.float()
        if self.num_classes is not None:
            self.label_embed = self.label_embed.float()

    def forward(
            self,
            x: th.Tensor,
            timesteps: th.Tensor,
            modalities: Dict[str, th.Tensor],
            y: Optional[th.Tensor] = None
    ) -> th.Tensor:
        """
        前向传播
        x: 输入噪声 [B, C, D, H, W]
        timesteps: 时间步 [B]
        modalities: 多模态输入 {模态名称: 特征张量}
        y: 类别标签 [B] (可选)
        """
        # 验证输入
        assert y is None or (self.num_classes is not None), "模型不支持类别条件"

        # 编码多模态特征
        modal_feat, encoded_modals = self.modal_encoder(modalities)

        # 时间嵌入
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # 融合类别嵌入
        if self.num_classes is not None and y is not None:
            emb = emb + self.label_embed(y)

        # 融合多模态特征
        emb = emb + modal_feat

        # 输入处理
        h = self.input_proj(x.type(self.dtype))
        hs = [h]

        # 下采样路径
        for module in self.down_blocks:
            if isinstance(module[0], PropertyAwareBlock):
                # 属性感知模块需要额外属性特征
                prop_feat = encoded_modals.get('property', modal_feat.unsqueeze(1))
                if prop_feat.dim() == 2:
                    prop_feat = prop_feat.unsqueeze(1)  # [B, 1, P]
                h = module(h, prop_feat)
            else:
                h = module(h, emb)
            hs.append(h)

        # 中间块
        h = self.middle_block(h, emb)

        # 上采样路径
        for module in self.up_blocks:
            if isinstance(module[0], PropertyAwareBlock):
                prop_feat = encoded_modals.get('property', modal_feat.unsqueeze(1))
                if prop_feat.dim() == 2:
                    prop_feat = prop_feat.unsqueeze(1)
                h = module(h, prop_feat)
            else:
                h = th.cat([h, hs.pop()], dim=1)
                h = module(h, emb)

        # 输出
        h = h.type(x.dtype)
        return self.out(h)


# 工具函数
def timestep_embedding(timesteps: th.Tensor, dim: int, max_period: int = 10000) -> th.Tensor:
    """生成时间步嵌入"""
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


# 模型使用示例
if __name__ == "__main__":
    # 配置多模态输入规格
    modal_specs = {
        'image': 3,  # RGB图像
        'property': 10,  # 10维属性特征
        'structure': 1  # 结构模态（如CT切片）
    }

    # 创建模型
    model = MultiModalDiffusionUNet(
        image_size=64,
        in_channels=1,
        model_channels=64,
        out_channels=1,
        num_res_blocks=2,
        attention_resolutions=(16, 8),
        channel_mult=(1, 2, 4),
        dims=3,
        num_classes=5,
        modal_specs=modal_specs,
        prop_channels=10
    )

    # 生成测试输入
    batch_size = 2
    x = th.randn(batch_size, 1, 64, 64, 64)  # 3D输入
    timesteps = th.randint(0, 1000, (batch_size,))
    modalities = {
        'image': th.randn(batch_size, 3, 64, 64),  # 2D图像
        'property': th.randn(batch_size, 10),  # 属性特征
        'structure': th.randn(batch_size, 1, 64, 64, 64)  # 3D结构
    }
    y = th.randint(0, 5, (batch_size,))  # 类别标签

    # 前向传播
    output = model(x, timesteps, modalities, y)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")