"""sub modules for composing ViT"""
import torch
from torch import nn


class PatchEmbedding(nn.Module):
    """small patches embedding
    image(B, C, H, W) -> projection(B, emb_dims, H/P, W/P) -> flatten & transpose(B, {(H/P) * (W/P)}, embed_dims)
    
    """
    
    def __init__(self, image_size=224, patch_size=16, in_channels=3, embed_dims=768, norm_layer=None, flatten=True):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        self.num_patches = (self.image_size[0] // self.patch_size[0]) * (self.image_size[1] // self.patch_size[1]) # N = HW / P^2
        self.flatten = flatten
        
        self.projection = nn.Conv2d(in_channels, embed_dims, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm_layer = norm_layer(embed_dims) if norm_layer else nn.Identity()
        
    def forward(self, x):
        x = self.projection(x) # B x embed_dims x (H/P) x (W/P)
        if self.flatten:
            x = torch.flatten(x, start_dim=2, end_dim=-1) # B x embed_dims x {(H/P) * (W/P)} = B x embed_dims x N
            x = torch.transpose(x, 1, 2) # B x N x embed_dims
        x = self.norm_layer(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head self attention layer"""
    
    def __init__(self, in_features, num_heads=8, qkv_bias=False, attention_drop=0., proj_drop=0.):
        """
        Args:
            in_features (int): input dimension
            num_heads (int, optional): [description]. Defaults to 8.
            qkv_bias (bool, optional): [description]. Defaults to False.
            attention_drop ([type], optional): [description]. Defaults to 0..
            proj_drop ([type], optional): [description]. Defaults to 0..
        """
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        head_dims = in_features // num_heads
        self.scale = head_dims ** -0.5 # ?
        
        self.qkv = nn.Linear(in_features, in_features * 3, bias=qkv_bias) # query, key, value
        self.attention_drop = nn.Dropout(attention_drop)
        self.projection = nn.Linear(in_features, in_features)
        self.projection_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        """for iterating self attention, output shape must be equal to input shape"""
        B, N, C = x.shape # output shape of patch-embedding layer, C : embedding dims
        qkv = self.qkv(x) # (B, N, 3*C)
        qkv = qkv.view(B, N, 3, self.num_heads, C // self.num_heads) # (B, N, 3, heads, C // heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous() # (3, B, heads, N, C // heads)
        query, key, value = qkv
        
        # scale dot-product attention in `Attention Is All You Need`
        attn_out = torch.matmul(query, key.transpose(-2, -1)) # matmul(B, heads, N, N)
        attn_out *= self.scale # scale
        attn_out = torch.softmax(attn_out, dim=-1)
        attn_out = self.attention_drop(attn_out)
        attn_out = torch.matmul(attn_out, value) # matmul(B, heads, N, C // heads)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C) # matmul(B, N, C)
        # attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C) # matmul(B, N, C)
        
        out = self.projection(attn_out)
        out = self.projection_drop(out)
        return out


class MLP(nn.Module):
    """MLP"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, activation_layer=nn.GELU, drop_rate=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = activation_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(drop_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
