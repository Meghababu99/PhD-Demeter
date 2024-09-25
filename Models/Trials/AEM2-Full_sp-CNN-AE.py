# %%
# %%
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
import tqdm
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from torch.utils.data import WeightedRandomSampler
from torchvision.utils import make_grid

import random
print(' AEM2_7 AUTOENCODER WITH FULL SPECTRUM INPUT(lr - 0.001, BS=64, epoch - 550)')
logging.warning(' AEM2_7 AUTOENCODER WITH FULL SPECTRUM INPUT(lr - 0.001, BS=64, epoch - 550)')

# %%
torch.manual_seed(21894)
np.random.seed(21894)
#  configuring device
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print('Running on the GPU')
  logging.warning(' Running on the GPU')
else:
  device = torch.device('cpu')
  print('Running on the CPU')
  logging.warning(' Running on the CPU')
# %%


class OrbitsDataset(Dataset):
    def __init__(self, pickle_dir, input_orbits,transform=None):
        self.transform = transform
        self.pickle_dir = pickle_dir
        self.input_orbits = input_orbits
        # self.file_list = [f for f in os.listdir(self.pickle_dir) if f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        self.fn = self.input_orbits.iloc[idx]['sp_filename']
        self.orbit = self.fn.replace('.pkl', '_full.pkl')
        self.data = pd.read_pickle(os.path.join(self.pickle_dir, self.orbit))
        self.sp0 = np.array([np.array(x) for x in self.data['spectrum0'][:400]]).astype(float)
        self.sp1 = np.array([np.array(x) for x in self.data['spectrum1'][:400]]).astype(float)
        self.x = np.empty((self.sp0.shape[0], self.sp0.shape[1] * 2))
        for i in range(self.sp0.shape[0]):
            self.x[i] = np.concatenate((self.sp0[i], self.sp1[i]))

        # Reshape x to have shape ( 800, 1024)
        self.x = self.x.reshape(( 800, 1024))
    
        if self.transform is not None:
            self.x = self.transform(self.x)

        return torch.tensor(self.x, dtype=torch.float32)

# # Example usage:
# pickle_dir = "G:\Ascii-ICE-phil"
# input_orbits = pd.read_csv("G:\ICE-CSV\phil_Clean-orbits.csv")
# dataset = OrbitsDataset(pickle_dir,input_orbits)

# # # Accessing a sample
# # sample_data = dataset.__getitem__(0)
# # print("Sample data shape:", sample_data.shape)



# Example usage:
pickle_dir = "/storage3/DSIP/Demeter/Ascii-ICE-phil"
input_orbits = pd.read_csv("/storage3/DSIP/Demeter/ICE/phil_Clean-orbits.csv")
dataset = OrbitsDataset(pickle_dir,input_orbits)

train_set, test_set = train_test_split(dataset.input_orbits, test_size=0.2, random_state= 42, shuffle = True)
logging.warning('train-test split done')


# %%
# val_set, test_set = train_test_split(test_set, test_size = 0.35, random_state =42, shuffle = True)
# logging.warning('Val-test split done')
# %%
print('length of trainset:',len(train_set))
# print('length of valset:',len(val_set))
print('length of testset:',len(test_set))
logging.warning('len of trainset is %s, test set is %s, str(len(train_set)),str(len(test_set))')
# %%


# %%
train_dataset = OrbitsDataset(pickle_dir, train_set)
# val_dataset = OrbitsDataset(pickle_dir, val_set)
test_dataset = OrbitsDataset(pickle_dir, test_set)


logging.warning('applying the  scaler')
# Create a StandardScaler object
scaler = StandardScaler()

# Fit the scaler on the training dataset
train_data = [data for data in train_dataset]
train_data = torch.cat(train_data, dim=0).numpy()
scaler.fit(train_data.flatten().reshape(-1, 1))

logging.warning('train_data scaler fit')
# Apply the same scaler to train dataset

# %%

scaled_train_data = []
for data in train_dataset:
    x = data.numpy().flatten().reshape(-1,1) 
    x = scaler.transform(x)
    x = x.reshape(800,1024)
    scaled_train_data.append((torch.from_numpy(x)))

# scaled_val_data = []
# for data in val_dataset:
#     x = data.numpy().flatten().reshape(-1,1) 
#     x = scaler.transform(x) 
#     x = x.reshape(390,2048)
#     scaled_val_data.append((torch.from_numpy(x)))

scaled_test_data = []
for data in test_dataset:
    x = data.numpy().flatten().reshape(-1,1)   
    x = scaler.transform(x)  
    x = x.reshape(800,1024)
    scaled_test_data.append((torch.from_numpy(x)))

# Dataloaders with scaled data
train_data_loader = DataLoader(scaled_train_data, batch_size=64, shuffle=True)
# val_data_loader = DataLoader(scaled_val_data, batch_size=18, shuffle=True)
test_data_loader = DataLoader(scaled_test_data, batch_size=64)
# logging.warning('Dataloading Done')
# %%

# %%


import torch.nn as nn
#  defining encoder
#  defining encoder
class Encoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=3, latent_dim=128, act_fn=nn.ReLU()):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 4,stride = 2,padding =1), 
        act_fn,
        # nn.Conv2d(out_channels, out_channels, 3,stride = 2), 
        # act_fn,
        nn.Conv2d(out_channels, 2*out_channels, 4, stride=2,padding =1), 
        act_fn,
        nn.Conv2d(2*out_channels, 4*out_channels, 4,stride = 2,padding =1),
        act_fn,
        nn.Conv2d(4*out_channels, 4*out_channels, 4, stride=2,padding =1), 
        # act_fn,
        # nn.Conv2d(4*out_channels, 4*out_channels, 4,stride = 2),
        act_fn,
        nn.Flatten(),
        nn.Linear(4*out_channels*50*64, latent_dim),
        act_fn
    )

  def forward(self, x):
    x =x.unsqueeze(1)
    print(x.shape)
    x = x.view(-1, 1,800,1024)
    print(x.shape)
    output = self.net(x)
    print(output.shape)
    return output


#  defining decoder
class Decoder(nn.Module):
  def __init__(self, in_channels=1, out_channels=3, latent_dim=128, act_fn=nn.ReLU()):
    super().__init__()

    self.out_channels = out_channels

    self.linear = nn.Sequential(
        nn.Linear(latent_dim, 4*out_channels*50*64),
        act_fn
    )

    self.conv = nn.Sequential(
        nn.ConvTranspose2d(4*out_channels, 4*out_channels, 4,stride = 2,padding =1), # (8, 8)
        act_fn,
        nn.ConvTranspose2d(4*out_channels, 2*out_channels, 4,stride = 2,padding =1), 
        act_fn,
        # nn.ConvTranspose2d(2*out_channels, 2*out_channels, 4,stride = 2),
        act_fn,
        nn.ConvTranspose2d(2*out_channels, out_channels, 4,stride = 2,padding =1), # 
        act_fn,
        # nn.ConvTranspose2d(out_channels, out_channels, 4,stride = 2),
        act_fn,
        nn.ConvTranspose2d(out_channels, in_channels, 4,stride = 2,padding =1)
    )

  def forward(self, x):
    output = self.linear(x)
    output = output.view(-1, 4*self.out_channels, 50,64)
    output = self.conv(output)
    return output


#  defining autoencoder
class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    # self.encoder.to(device)

    self.decoder = decoder
    # self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded






# %%
encoder = Encoder()
decoder = Decoder()
model = Autoencoder(encoder, decoder)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%

num_epochs = 550
train_loss=[]
test_loss = []
for epoch in range(num_epochs):
  train_epoch_loss = 0
  test_epoch_loss = 0
  model.train()
  for i, data in enumerate(train_data_loader):
    inputs = data
    outputs = model(inputs)
    loss = loss_fn(outputs, inputs)
    
    train_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
    # print(i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if i == 100:
    #   break
  train_epoch_loss = train_epoch_loss /(i+1)
  train_loss.append(train_epoch_loss)
  model.eval()
  for j, data in enumerate(test_data_loader):
    inputs = data
    outputs = model(inputs)
    loss = loss_fn(outputs, inputs)
    test_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
  test_epoch_loss = test_epoch_loss /(j+1)
  test_loss.append(test_epoch_loss)

  print('Epoch: {} \tTraining Loss: {:.6f} \ttest Loss: {:.6f}'.format(epoch, train_epoch_loss, test_epoch_loss))
torch.save(model.state_dict(), 'trained_model-AEM2_7.pth')
# %%

# Get the encoded representation of your data
encoded_train_data = []
with torch.no_grad():
    for traindata in train_data_loader:
        encoded_train = model.encoder(traindata)
        encoded_train_data.append(encoded_train.numpy())

encoded_train = np.concatenate(encoded_train_data, axis=0)


encoded_test_data = []
with torch.no_grad():
    for data1 in test_data_loader:
        encoded_test = model.encoder(data1)
        encoded_test_data.append(encoded_test.numpy())

encoded_test = np.concatenate(encoded_test_data, axis=0)
plt.figure(figsize=(10,8))
# Plot the encoded data distribution for training data
plt.scatter(encoded_train[:, 0], encoded_train[:, 1], cmap='viridis', label='Train')

# Plot the encoded data distribution for testing data
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], cmap='viridis',label='Test')
 
plt.axvline(0, c='black', ls='--')

# Adding horizontal line in data co-ordinates
plt.axhline(0, c='black', ls='--')

# plt.xlabel('Node 1')
# plt.ylabel('Node 2')
plt.legend()
plt.title('Encoded data [batch size =64, lr =0.001,epoch =550,LAT_DIM =128 ]')
plt.savefig('Encoded_ICE-AEM2_7.png')

# %%
# %%
plt.figure(figsize=(10,8))
plt.plot(train_loss,label ='train loss')
plt.plot(test_loss,label = 'test loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and test Loss [batch size = 64, lr =0.001,LAT_DIM =128]')
# plt.title('AEM1_0 [batch size = 8, lr =0.001 ]')
plt.savefig('Test-train_ICE-AEM2_7.png')

# %%
all_test_images = []
all_reconstructed_imgs = []

with torch.no_grad():
    for test_images in test_data_loader:
        # Sending test images to device
        test_images = test_images.to(device)
        
        # Reconstructing test images
        reconstructed_imgs = model(test_images)
        
        # Appending current batch outputs to the list
        all_test_images.append(test_images.cpu())
        all_reconstructed_imgs.append(reconstructed_imgs.cpu())

# Concatenating outputs of all batches along the batch dimension
all_test_images = torch.cat(all_test_images, dim=0)
all_reconstructed_imgs = torch.cat(all_reconstructed_imgs, dim=0)

# Visualization
imgs = torch.stack([all_test_images.view(-1, 1, 800, 1024), all_reconstructed_imgs], dim=1).flatten(0, 1)
grid = make_grid(imgs, nrow=8, normalize=True, padding=1)
grid = grid.permute(1, 2, 0)

plt.figure(dpi=170)
plt.title('Original/Reconstructed')
plt.imshow(grid)
plt.axis('off')
plt.savefig('Original-Reconstructed-AEM2_7.png')

# for test_images in test_data_loader:
#         #  sending test images to device
#     test_images = test_images.to(device)
#     with torch.no_grad():
#           #  reconstructing test images
#         reconstructed_imgs = model(test_images)
#         #  sending reconstructed and images to cpu to allow for visualization
#         reconstructed_imgs = reconstructed_imgs.cpu()
#         test_images = test_images.cpu()

#         #  visualisation
#         imgs = torch.stack([test_images.view(-1,1,800,1024), reconstructed_imgs], 
#                           dim=1).flatten(0,1)
#         grid = make_grid(imgs, nrow=8, normalize=True, padding=1)
#         grid = grid.permute(1, 2, 0)
#         plt.figure(dpi=170)
#         plt.title('Original/Reconstructed')
#         plt.imshow(grid)
#         # log_dict['visualizations'].append(grid)
#         plt.axis('off')
#         plt.savefig('Original-Reconstructed-AEM2_6.png')

# %%



