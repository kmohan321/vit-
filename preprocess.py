import torch
import torch.nn as nn


'''function for patching the images'''
class Patch(nn.Module):
  def __init__(self,in_channels,patch_size,embed_dim):
    super().__init__()

    self.patch_size = patch_size
    self.embed_dim = embed_dim
    # num_patches = (image_height//patch_size)**2
    self.patch_embed = nn.Conv2d(in_channels,embed_dim,kernel_size=patch_size,stride=patch_size)

  def forward(self,x):
    # (batch,channels,height,width) -> (batch,embed_dim,height//patch_size,width//patch_size)
    x = self.patch_embed(x)
    # (batch,embed_dim,height//patch_size,width//patch_size) -> (batch,embed_dim,height//patch_size*width//patch_size)   
    x = x.flatten(2)
    '''here it is important to know that transformer accepts inputs as (batch,num_patches,embed_dim)'''
    x = x.transpose(1,2)
    return x

'''function for adding the pos_embedding and the class token to patch tokens'''
class Pos_embedding(nn.Module):
  def __init__(self,embed_dim,num_patches):
    super().__init__()
    self.embed_dim = embed_dim
    self.num_patches = num_patches
    '''these are learnable parameters as mentioned in the paper of vit '''
    self.pos_embed = nn.Parameter(torch.randn(1,num_patches+1,embed_dim))
    self.class_token = nn.Parameter(torch.randn(1,1,embed_dim))

  def forward(self,x):
    batch_size = x.shape[0]
    class_token_expanded = self.class_token.expand(batch_size,-1,-1)
    # added the class embedding to patch embeddings
    x = torch.cat((class_token_expanded,x),dim=1)
    # added the positional embedding to the patch embeddings
    x = x + self.pos_embed
    return x
