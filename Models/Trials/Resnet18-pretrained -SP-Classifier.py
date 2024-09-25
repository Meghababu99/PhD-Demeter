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
from torchvision import models

print('output of Resnet18-pretrained (lr - 0.0006, BS= 12, epoch - 200)')
logging.warning('output of Resnet18-pretrained (lr - 0.0006, BS= 12, epoch - 200)')
# %%
class OrbitsDataset(Dataset):
    def __init__(self, pickle_dir,input_orbits, transform=None):
        self.transform = transform
        self.pickle_dir = pickle_dir
        self.input_orbits = input_orbits

    def __len__(self):
        return len(self.input_orbits)

    def __getitem__(self, idx):
        try:
            self.up_orb_fn = self.input_orbits.iloc[idx]['UpOrbits']
            self.dn_orb_fn = self.input_orbits.iloc[idx]['DownOrbits']
            self.up_orb = pd.read_pickle(os.path.join(self.pickle_dir, self.up_orb_fn))
            self.up_orb = torch.from_numpy(self.up_orb)[0:410]
            self.dn_orb = pd.read_pickle(os.path.join(self.pickle_dir, self.dn_orb_fn))
            self.dn_orb = torch.from_numpy(self.dn_orb)[0:410]
            self.x = torch.cat([self.up_orb, self.dn_orb], 0)

            self.y = self.input_orbits.iloc[idx]['Labels']

            if self.transform is not None:
                self.x = self.transform(self.x)
            return self.x, self.y

        except IndexError:
            raise IndexError(f"Index {idx} is out of range.")



folder_path = '/storage3/DSIP/Demeter/ASCII_1132'
input_orbits = pd.read_csv('/storage3/DSIP/Demeter/ICE/Balanced.csv')
logging.warning('the Orbitdataset class has covered')

dataset = OrbitsDataset(folder_path, input_orbits,transform=None)
print(len(dataset))

train_set, test_set = train_test_split(dataset.input_orbits, test_size=0.20, random_state= 42, shuffle =False)
logging.warning('train-test split done')
val_set, test_set = train_test_split(test_set, test_size = 0.50, random_state =42, shuffle = True)
logging.warning('Val-test split done')
# %%
print('length of trainset:',len(train_set))
print('length of valset:',len(val_set))
print('length of testset:',len(test_set))
logging.warning('len of trainset is %s, test set is %s, val set is %s', str(len(train_set)),str(len(test_set)),str(len(val_set)))
# %%
train_dataset = OrbitsDataset(folder_path, train_set)
val_dataset = OrbitsDataset(folder_path, val_set)
test_dataset = OrbitsDataset(folder_path, test_set)

logging.warning('applying the  scaler')
# Create a StandardScaler object
scaler = StandardScaler()

logging.warning('fit scaler')
# Fit the scaler on the training dataset
train_data = [data for data, _ in train_dataset]
train_data = torch.cat(train_data, dim=0).numpy()
scaler.fit(train_data.flatten().reshape(-1, 1))

logging.warning('train_data scaler fit')
# Apply the same scaler to train dataset

scaled_train_data = []
for data, target in train_dataset:
    x = data.numpy().flatten().reshape(-1,1) 
    x = scaler.transform(x)
    x = x.reshape(820,1024)
    scaled_train_data.append((torch.from_numpy(x), target))

scaled_val_data = []
for data, target in val_dataset:
    x = data.numpy().flatten().reshape(-1,1) 
    x = scaler.transform(x) 
    x = x.reshape(820,1024)
    scaled_val_data.append((torch.from_numpy(x), target))

scaled_test_data = []
for data, target in test_dataset:
    x = data.numpy().flatten().reshape(-1,1)   
    x = scaler.transform(x)  
    x = x.reshape(820,1024)
    scaled_test_data.append((torch.from_numpy(x), target))

# Dataloaders with scaled data
train_data_loader = DataLoader(scaled_train_data, batch_size=12, shuffle=True)
val_data_loader = DataLoader(scaled_val_data, batch_size=12, shuffle=True)
test_data_loader = DataLoader(scaled_test_data, batch_size=12)
logging.warning('Dataloading Done')

# %%
logging.warning('approching Model class')
# class ResNetModel(nn.Module):
#     def __init__(self, num_classes=1):
#         super(ResNetModel, self).__init__()
#         # Load a pre-trained ResNet model (e.g., ResNet-50)
#         resnet = models.resnet50(pretrained=True)
#          # Replace the first convolutional layer to accept a single channel
#         resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
        
#         # Remove the final classification layer
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules)
        
#         # Add your custom classification layers
#         self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.linear1 = nn.Linear(2048,num_classes)
#         # self.linear2 = nn.Linear(1024,512)
#         # self.linear3 = nn.Linear(512,num_classes)
#         # self.linear4 = nn.Linear(256,50)
#         # self.linear5 = nn.Linear(50,num_classes)
#         # self.maxpool = nn.MaxPool2d(2,2)
#         # self.relu = nn.ReLU()
#         # self.dropout = nn.Dropout(0.4)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.resnet(x)
#         x = self.avg_pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.linear1(x)
#         # x = self.relu(x)
#         # x = self.dropout(x)
#         # x = self.linear2(x)
#         # x = self.relu(x)
#         # x = self.dropout(x)
#         # x = self.linear3(x)
#         x = self.sigmoid(x)
#         return x
    #     # Replace the first convolutional layer to accept a single channel
    #     resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    #     # Remove the final classification layer
    #     modules = list(resnet.children())[:-2]
    #     self.resnet = nn.Sequential(*modules)
        
    #     # Add your custom classification layers
    #     self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
    #     self.fc = nn.Linear(2048, num_classes)  # Adjust the output dimension if needed
    #     self.sigmoid = nn.Sigmoid()

    # def forward(self, x):
    #     x = self.resnet(x)
    #     x = self.avg_pool(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     x = self.sigmoid(x)
    #     return x

# Instantiate the model
# model = ResNetModel()
def modify_resnet(model):
    # Freeze all layers except the first convolutional layer
    # for name, param in model.named_parameters():
        # if 'conv1' not in name:
        #     param.requires_grad = False
    for param in model.conv1.parameters():
        param.requires_grad = False

    # Freeze layers in the "layer1" block
    for param in model.layer1.parameters():
        param.requires_grad = False
    for param in model.layer2.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
    model.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, bias=False)
    # Replace the output layers with a fully connected layer followed by a sigmoid
    model.fc = nn.Sequential(
        nn.Linear(512, 1),  # Modify 'your_hidden_units' as needed
        # nn.ReLU(),  # Optional activation function
        # nn.Linear(1024,1),  # Modify 'your_output_units' as needed
        nn.Sigmoid()
    )
model = models.resnet18(pretrained=True)

# %%

loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.0006)
print('model summary', model)

# %%
num_epoch=200
train_loss=[]
val_loss = []
best_val_loss = 10  # Initialize with a large value
patience = 60 # Number of epochs without improvement to wait before stopping
counter = 0
for epoch in range(num_epoch):
  train_epoch_loss=0
  val_epoch_loss=0
  model.train()
  for i, (data, target) in enumerate(train_data_loader):
    modify_resnet(model)
    data =  data.unsqueeze(1)
    
    output=model(data.float())
    loss=loss_fn(output,target.unsqueeze(1).float())
    train_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
    # print(i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # if i == 70:
    #    break 
  print(list(model.layer1.parameters())[0])
  train_epoch_loss = train_epoch_loss /(i+1)
  train_loss.append(train_epoch_loss)
  model.eval()
  for i, (data, target) in enumerate(val_data_loader):
    data =  data.unsqueeze(1)
    output=model(data.float())
    loss=loss_fn(output,target.unsqueeze(1).float())
    val_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
  val_epoch_loss = val_epoch_loss /(i+1)
  val_loss.append(val_epoch_loss)

  print('Epoch: {} \tTraining Loss: {:.6f} \tval Loss: {:.6f}'.format(epoch, train_epoch_loss, val_epoch_loss))
  logging.warning('epoch %s, the training loss is %s and validation loss is %s', str(epoch), str(train_epoch_loss), str(val_epoch_loss))
  if val_epoch_loss < best_val_loss:
    best_val_loss = val_epoch_loss
    counter = 0
    print('best loss',best_val_loss )
    # torch.save(model.state_dict(), 'best_model.pth')
  else:
    counter += 1
    if counter >= patience:
        print("Early stopping triggered.")
        logging.warning("Early stopping triggered.")
        break

# %%
model.eval()
y_truet= []
y_predt =[]
for i, (data, target) in enumerate(train_data_loader):
    data = data.unsqueeze(1)
    output=model(data.float())
    y_predt.append(output.detach().numpy())
    y_truet.append(target.detach().numpy())
y_predt = np.concatenate(y_predt).squeeze()
y_truet = np.concatenate(y_truet)

binary_predictions = (y_predt >= 0.5).astype(int)
# Calculate the confusion matrix
conf_matrix = sklearn.metrics.confusion_matrix(y_truet, binary_predictions)
print("Confusion Matrix FOR TRAIN SET:")
print(conf_matrix)

# Calculate the accuracy
accuracy = sklearn.metrics.accuracy_score(y_truet, binary_predictions)
print(f"Accuracy for training set: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_truet, np.round(y_predt))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion matrics_train')
plt.savefig('Confusion matrics_train_pretrained_1.png')

# %%
fig = plt.figure(figsize=(10,8))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(frameon=False)
plt.title('Balanced dataset [batch size =12, lr =0.0006]')
plt.savefig('Test-train-pretrained_1.png')
plt.show()

# %%
model.eval()
y_true = []
y_pred =[]
for i, (data, target) in enumerate(test_data_loader):
    data = data.unsqueeze(1)
    output=model(data.float())
    y_pred.append(output.detach().numpy())
    y_true.append(target.detach().numpy())
y_pred = np.concatenate(y_pred).squeeze()
y_true = np.concatenate(y_true)

print('The true y', y_true)
print('The predicted y', y_pred)

plt.figure(figsize=(10,8))
plt.hist(y_pred[y_true==0], bins=200, alpha=0.5, label='Non_seismic')
plt.hist(y_pred[y_true==1], bins=200, alpha=0.5, label='seismic')
plt.legend()
plt.xlabel('Disciminator output')
plt.show()
plt.savefig('Disciminator output_pretrained_1.png')

# Convert the continuous output to binary predictions using a threshold of 0.5
binary_predictions = (y_pred >= 0.5).astype(int)
# %%
# Calculate the confusion matrix
conf_matrix = sklearn.metrics.confusion_matrix(y_true, binary_predictions)

print("Confusion Matrix:")
print(conf_matrix)

# Calculate the accuracy
accuracy = sklearn.metrics.accuracy_score(y_true, binary_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# %%
print('Accuracy: {:.6f}'.format(accuracy_score(np.round(y_pred), y_true)))


cm = confusion_matrix(y_true, np.round(y_pred))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
# plt.legend(frameon=False)
plt.show()
plt.savefig('Confusion matrics_pretrained_1.png')
# plot ROC curve

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.01,1.0])
plt.ylim([0.0,1.01])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right', frameon=False)
plt.show()
plt.savefig('ROC_pretrained_1.png')





