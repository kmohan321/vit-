import torch
from vit_model import VIT
from preprocess import Patch,Pos_embedding
from torch.utils.data import DataLoader
from data import data_train,data_test
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
writer  = SummaryWriter()
from torch.optim.lr_scheduler import CosineAnnealingLR

'''HYPERPARAMETRS'''
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
embedding_dim = 256
image_height = 32
num_heads = 4
num_layers = 6
mlp_size = 512
num_classes = 10
in_channels = 3
patch_size = 4
batch_size = 128
lr = 0.0001
num_patches = (image_height//patch_size)**2
load_model = False


vit  = VIT(embedding_dim,num_heads,num_layers,mlp_size,patch_size,in_channels,num_patches,num_classes).to(device)
if load_model ==True:
  vit = torch.load('vit.pth')
  print('load successfully')

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(),lr=lr)

dataloader_train = DataLoader(data_train,batch_size=batch_size,shuffle=True)
dataloader_test = DataLoader(data_test,batch_size=batch_size,shuffle=False)


global_steps = 0
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
for epoch in range(epochs):
  vit.train()
  avg_loss = 0
  for batch_idx,(data,targets) in tqdm(enumerate(dataloader_train)):
    global_steps += 1

    data = data.to(device)
    # if len(data)<batch_size:
    #   continue
    targets = targets.to(device)
    output = vit(data)
    loss = loss_fn(output,targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item()
    writer.add_scalar('loss',loss.item(),global_steps)
  scheduler.step()
  print(f'Epoch: {epoch+1}/{epochs} || Loss: {avg_loss/epochs:.4f}')
  torch.save(vit,'vit.pth')
  # model evaluation 
  images,target = next(iter(dataloader_test))
  images,target = images.to(device),target.to(device)
  vit.eval()
  with torch.no_grad():
    outputs = vit(images)
  predicted_label = torch.argmax(torch.softmax(outputs,dim=1),dim=1).squeeze(0)
  # print(predicted_label)
  # print(target)
  matches = (predicted_label==target).sum().item()
  # print(matches)
  print(f'accuracy {matches/len(target):.4f}')


    
