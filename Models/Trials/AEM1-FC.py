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
from tqdm import tqdm
import sklearn
# import seaborn as sns
logging.warning(' AEM1_3 New-AUTOENCODER WITH [11,2](min,max) INPUT(lr - 0.001, BS=64, epoch - 300)')
# %%
torch.manual_seed(21894)
np.random.seed(21894)

# %%


class CustomDataset(Dataset):
    def __init__(self, data_frame):
        self.data_frame = data_frame

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        self.data = self.data_frame.iloc[idx]['features']
        self.features = np.array(self.data)[:,:56]
        self.result_array = []
        for sublist in self.features:
            self.first_value = sublist.max()
            self.last_value = sublist.min()
            self.result_array.append((self.first_value, self.last_value))
        self.flattened_array = [item for sublist in self.result_array for item in sublist]
        self.label = self.data_frame.iloc[idx]['label']
        self.features_tensor = torch.tensor(self.flattened_array, dtype=torch.float32)
        return self.features_tensor , self.label

class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),  # Corrected: Added parentheses after nn.ReLU
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),  # Corrected: Added parentheses after nn.ReLU
            nn.Linear(32, 18),
            nn.ReLU(),
            nn.Linear(18, 6),
            nn.ReLU(),
            nn.Linear(6, 2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(2, 6),
            nn.ReLU(),
            nn.Linear(6, 18),
            nn.ReLU(),
            nn.Linear(18, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_size),  # Adjust input size based on the original input size
            nn.ReLU()  # You can use a different activation function if required
        )


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
logging.warning('the Orbitdataset class AND MODEL has covered')
# %%
data_frame = pd.read_pickle("/storage3/DSIP/Demeter/Newdataset/Phil_D2_ICE-120.pkl")
custom_dataset = CustomDataset(data_frame)
# %%
train, test = train_test_split(custom_dataset.data_frame,test_size = 0.2, shuffle =True)

train_dataset = CustomDataset(train)
test_dataset = CustomDataset(test)
logging.warning('the Orbitdataset TRAIN_TEST SPLIT has covered')
# %%
scaler = StandardScaler()

# Fit the scaler on the training dataset
train_data = [data for data,_ in train_dataset]
train_data = torch.cat(train_data, dim=0).numpy()
scaler.fit(train_data.flatten().reshape(-1, 1))

# %%
scaled_train_data = []
for data,target in train_dataset:
    x = data.numpy().flatten().reshape(-1,1) 
    x = scaler.transform(x)
    x = x.reshape(22)
    scaled_train_data.append((torch.from_numpy(x),target))

scaled_test_data = []
for data,target in test_dataset:
    x = data.numpy().flatten().reshape(-1,1)   
    x = scaler.transform(x)  
    x = x.reshape(22)
    scaled_test_data.append((torch.from_numpy(x),target))

# %%
train_data_loader = DataLoader(scaled_train_data, batch_size=64, shuffle=True)
# val_data_loader = DataLoader(scaled_val_data, batch_size=6, shuffle=True)
test_data_loader = DataLoader(scaled_test_data, batch_size=64)

# %%
input_size = 22
model = Autoencoder(input_size)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# %%


num_epochs = 300
train_loss=[]
test_loss = []
for epoch in range(num_epochs):
  train_epoch_loss = 0
  test_epoch_loss = 0
  model.train()
  for i, (data,target) in enumerate(train_data_loader):
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
  for j, (data,target) in enumerate(test_data_loader):
    inputs = data
    outputs = model(inputs)
    loss = loss_fn(outputs, inputs)
    test_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
  test_epoch_loss = test_epoch_loss /(j+1)
  test_loss.append(test_epoch_loss)

  print('Epoch: {} \tTraining Loss: {:.6f} \ttest Loss: {:.6f}'.format(epoch, train_epoch_loss, test_epoch_loss))


torch.save(model.state_dict(), 'trained_model-AEM1_3.pth')

# %%
X_test_data = np.array([sample[0].numpy() for sample in scaled_test_data])
y_test = np.array([sample[1] for sample in scaled_test_data])

reconstructed_test_data = []
with torch.no_grad():
    for data, _ in test_data_loader:
        reconstructed = model.encoder(data)
        reconstructed_test_data.append(reconstructed.numpy())
reconstructed_test_data = np.concatenate(reconstructed_test_data, axis=0)


print("Shape of X_test_data:", X_test_data.shape)
print("Shape of y_test:", y_test.shape)

X_test_label_0 = X_test_data[y_test == 0]
X_test_label_1 = X_test_data[y_test == 1]
plt.figure(figsize=(10,8))
# # Plotting reconstructed data points with label 0 in light blue
plt.scatter(reconstructed_test_data[y_test == 0][:, 0], reconstructed_test_data[y_test == 0][:, 1], color='lightblue', label='non seismic')

# # Plotting reconstructed data points with label 1 in light orange
plt.scatter(reconstructed_test_data[y_test == 1][:, 0], reconstructed_test_data[y_test == 1][:, 1], color='lightsalmon', label='seismic')
plt.axvline(0, c='black', ls='--')

# Adding horizontal line in data co-ordinates
plt.axhline(0, c='black', ls='--')

plt.xlabel('Node 1')
plt.ylabel('Node 2')
plt.title('Reconstructed TEST data [batch size =64, lr =0.001, epochs = 300]')
plt.legend()
plt.savefig('TEST_Encoded_ICE-AEM1_3.png')

#%%
plt.figure(figsize=(10,8))
# Plotting original data points with label 0 in blue
plt.scatter(X_test_label_0[:, 0], X_test_label_0[:, 1], color='blue', label='non seismic')

# Plotting original data points with label 1 in orange
plt.scatter(X_test_label_1[:, 0], X_test_label_1[:, 1], color='orange', label='seismic')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('INPUT Test Data [batch size = 64, lr =0.001, epochs = 300]')
plt.legend()
plt.savefig('TEST_Input_ICE-AEM1_3.png')



#%%
X_train_data = np.array([sample[0].numpy() for sample in scaled_train_data])
y_train= np.array([sample[1] for sample in scaled_train_data])

reconstructed_train_data = []
with torch.no_grad():
    for data, _ in train_data_loader:
        reconstructed = model.encoder(data)
        reconstructed_train_data.append(reconstructed.numpy())
reconstructed_train_data = np.concatenate(reconstructed_train_data, axis=0)


print("Shape of X_test_data:", X_train_data.shape)
print("Shape of y_test:", y_train.shape)

X_train_label_0 = X_train_data[y_train== 0]
X_train_label_1 = X_train_data[y_train== 1]
plt.figure(figsize=(10,8))
# # Plotting reconstructed data points with label 0 in light blue
plt.scatter(reconstructed_train_data[y_train == 0][:, 0], reconstructed_train_data[y_train == 0][:, 1], color='lightblue', label='non seismic')

# # Plotting reconstructed data points with label 1 in light orange
plt.scatter(reconstructed_train_data[y_train == 1][:, 0], reconstructed_train_data[y_train == 1][:, 1], color='lightsalmon', label='seismic')
plt.axvline(0, c='black', ls='--')

# Adding horizontal line in data co-ordinates
plt.axhline(0, c='black', ls='--')

plt.xlabel('Node 1')
plt.ylabel('Node 2')
plt.title('Reconstructed Train data [batch size = 64, lr =0.001, epochs = 300]')
plt.legend()
plt.savefig('TRAIN_Encoded_ICE-AEM1_3.png')
# plt.show()
#%%

plt.figure(figsize=(10,8))
# Plotting original data points with label 0 in blue
plt.scatter(X_train_label_0[:, 0], X_train_label_0[:, 1], color='blue', label='non seismic')

# Plotting original data points with label 1 in orange
plt.scatter(X_train_label_1[:, 0], X_train_label_1[:, 1], color='orange', label='seismic')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('INPUT Train Data [batch size = 64, lr =0.001, epochs = 300]')
plt.legend()
plt.savefig('TRAIN_input_ICE-AEM1_3.png')
#%%

# plt.show()

# # Get the encoded representation of your data
# encoded_train_data = []
# with torch.no_grad():
#     for traindata in train_data_loader:
#         encoded_train = model.encoder(traindata)
#         encoded_train_data.append(encoded_train.numpy())

# encoded_train = np.concatenate(encoded_train_data, axis=0)


# encoded_test_data = []
# with torch.no_grad():
#     for data1 in test_data_loader:
#         encoded_test = model.encoder(data1)
#         encoded_test_data.append(encoded_test.numpy())

# encoded_test = np.concatenate(encoded_test_data, axis=0)
# plt.figure(figsize=(10,8))
# # Plot the encoded data distribution for training data
# plt.scatter(encoded_train[:, 0], encoded_train[:, 1], cmap='viridis', label='Train')

# # Plot the encoded data distribution for testing data
# plt.scatter(encoded_test[:, 0], encoded_test[:, 1], cmap='viridis',label='Test')
 
# plt.axvline(0, c='black', ls='--')

# # Adding horizontal line in data co-ordinates
# plt.axhline(0, c='black', ls='--')

# plt.xlabel('Node 1')
# plt.ylabel('Node 2')
# plt.legend()
# plt.title('Encoded data [batch size = 32, lr =0.0001, epochs = 300]')
# plt.savefig('Encoded_ICE-AEM1_4.png')

# %%
plt.figure(figsize=(10,8))
plt.plot(train_loss,label ='train loss')
plt.plot(test_loss,label = 'test loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and test Loss [batch size = 64, lr =0.001 ]')
# plt.title('AEM1_0 [batch size = 8, lr =0.001 ]')
plt.savefig('Test-train_ICE-new-AEM1_3.png')