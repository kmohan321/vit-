import torch
import torch.nn as nn
from torchsummary import summary
from preprocess import Patch,Pos_embedding
'''point to remember this model takes preprocess images as data 
that is patched images as tokens(llms tokens)'''

class VIT(nn.Module):
  def __init__(self,embed_dim,num_heads,layers,mlp_size,patch_size,in_channels,num_patches,num_classes):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_heads = num_heads
    self.layers = layers
    # self.hidden_size=hidden_size
    self.mlp_size = mlp_size
    # transformer block
    self.patch = Patch(in_channels,patch_size,embed_dim)
    self.pos_embedding = Pos_embedding(embed_dim,num_patches)
    self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dim_feedforward=mlp_size,activation='gelu',batch_first=True),\
      num_layers=layers)
    self.mlp = nn.Sequential(
        nn.LayerNorm(embed_dim),
        nn.Linear(embed_dim,num_classes),
    )
  def forward(self,x):
    patch_data  = self.patch(x)
    pos_data = self.pos_embedding(patch_data)
    x = self.transformer_encoder(pos_data)
    x = x[:,0,:]
    x = self.mlp(x)
    return x

vit = VIT(256,4,6,2048,4,3,49,10) 
summary(vit)
