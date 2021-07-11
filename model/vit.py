"""Vision Transformer"""
from collections import OrderedDict
import math
import torch
from torch import nn

from model.modules import PatchEmbedding, MultiHeadAttention, MLP, DropPath
from model.init_weight import trunc_normal, init_vit_weights


class VisionTransformer(nn.Module):
    
    def __init__(self, 
                 image_size=224,
                 patch_size=16, 
                 in_channels=3, 
                 num_classes=1000, 
                 embed_dims=768,
                 representation_size=None,
                 drop_rate=0.,
                 num_blocks=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 attn_drop_rate=0., 
                 drop_path_rate=0,
                 activation_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        Args:
            image_size (int, optional): input image size. Defaults to 224.
            patch_size (int, optional): patch size. Defaults to 16.
            in_channels (int, optional): input image channels. Defaults to 3.
            num_classes (int, optional): number of classes to train. Defaults to 1000.
            embed_dims (int, optional): patch embedding dimension. Defaults to 768.
            representation_size (int, optional): using another dimension except embed_dims. Defaults to None.
            drop_rate (float, optional): dropout probability. Defaults to 0..
            num_blocks (int, optional): number of Transformer blocks. Defaults to 12.
            num_heads (int, optional): number of heads. Defaults to 12.
            mlp_ratio (float, optional): feature scaling ratio used in MLP. Defaults to 4..
            qkv_bias (bool, optional): if set, qkv bias in Multi-Head Attention layer is to be considered. Defaults to True.
            attn_drop_rate (float, optional): dropout probability after each attention. Defaults to 0..
            drop_path_rate (int, optional): dropout rate for stochastic depth ratio. Defaults to 0.
            activation_layer (nn.Module, optional): activating function. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        
        self.patch_emb = PatchEmbedding(image_size, patch_size, in_channels, embed_dims, norm_layer)
        num_patches = self.patch_emb.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.position_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dims))
        self.position_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_blocks)]  # stochastic depth decay rule -> ?
        self.blocks = nn.Sequential(*[
            Block(in_features=embed_dims, 
                  num_heads=num_heads, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias,
                  drop_rate=drop_rate,
                  attn_drop_rate=attn_drop_rate,
                  drop_path=dpr[i],
                  activation_layer=activation_layer,
                  norm_layer=norm_layer
                  ) for i in range(num_blocks)
            ]
                                    )
        self.norm = norm_layer(embed_dims)
        
        # representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dims, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        
        # classifier heads
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        
        # initialize weights
        self.__weight_initialize()
        
    def forward(self, x):
        x = self.patch_emb(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # (B, 1, embed_dims)
        x = torch.cat([cls_token, x], dim=1) # (B, N+1, embed_dims)
        x = self.position_drop(x + self.position_embed) # add(&broadcast) position embedding parameters per example
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x)
        x = self.pre_logits(x[:, 0, :])
        return x
    
    def __weight_initialize(self):
        self.position_embed = trunc_normal(self.position_embed, std=0.02)
        self.cls_token = trunc_normal(self.cls_token, std=0.02)
        self.apply(init_vit_weights)
        
    def load_pretrained(self):
        """using pretrained weights"""
        pass


class Block(nn.Module):
    """
    Block is composed of multi-head attention & MLP(feedforward).
        (1) norm_layer 
        (2) multi-head attention 
        (3) shortcut 
        (4) norm_layer 
        (5) MLP 
        (6) shortcut
    It will be iterated several times
    """
    
    def __init__(self, 
                 in_features, 
                 num_heads, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 drop_rate=0., 
                 attn_drop_rate=0., 
                 drop_path=0.,
                 activation_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        """
        Args:
            in_features (int): input dimension
            num_heads (int): number of heads to use
            mlp_ratio (float, optional): hidden dimension size of MLP layer. Defaults to 4..
            qkv_bias (bool, optional): if using qkv hidden layer's bias. Defaults to False.
            drop_rate (float, optional): dropout ratio. Defaults to 0..
            attn_drop_rate (float, optional): dropout ratio in multi-head attention. Defaults to 0..
            drop_path (float, optional): ???. Defaults to 0..
            activation_layer (nn.Module, optional): activation function(layer). Defaults to nn.GELU.
            norm_layer (nn.Module, optional): normalization layer. Defaults to nn.LayerNorm.
        """
        super(Block, self).__init__()
        self.norm1 = norm_layer(in_features)
        self.multihead_attention = MultiHeadAttention(in_features, 
                                                      num_heads=num_heads,
                                                      qkv_bias=qkv_bias,
                                                      attention_drop=attn_drop_rate,
                                                      proj_drop=drop_rate)
        
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(in_features)
        
        mlp_hidden_features = int(in_features * mlp_ratio)
        self.mlp = MLP(in_features, 
                       hidden_features=mlp_hidden_features,
                       activation_layer=activation_layer,
                       drop_rate=drop_rate)
        
    def forward(self, x_in):
        x = self.norm1(x_in)
        x_in = x_in + self.drop_path(self.multihead_attention(x))
        
        x = self.norm2(x_in)
        x = x_in + self.drop_path(self.mlp(x))
        return x


if __name__ == '__main__':
    sample = torch.randint(0, 256, (1, 3, 224, 224)) / 255.
    
    model = VisionTransformer(num_classes=10)
    output = model(sample)
    print('output.shape : ', output.shape)
