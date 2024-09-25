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
print('output of model-ICE-D1-s (lr - 0.005, BS= 3, epoch - 60)')
logging.warning('output of model-ICE-D1-s (lr - 0.005, BS= 3, epoch - 60)')
# %%

torch.manual_seed(21894)
np.random.seed(21894)

class OrbitsDataset(Dataset):
    def __init__(self, pickle_dir, csv_file, transform=None):
        self.transform = transform
        self.pickle_dir = pickle_dir
        self.csv_file = csv_file
        self.input_orbits = csv_file

    def __len__(self):
        return len(self.input_orbits)

    def __getitem__(self, idx):
        try:
            self.up_orb_fn = self.input_orbits.iloc[idx]['UpOrbits']
            self.dn_orb_fn = self.input_orbits.iloc[idx]['DownOrbits']
            self.up_orb = pd.read_pickle(os.path.join(self.pickle_dir, self.up_orb_fn))
            self.up_orb = torch.from_numpy(self.up_orb)[0:390]
            self.dn_orb = pd.read_pickle(os.path.join(self.pickle_dir, self.dn_orb_fn))
            self.dn_orb = torch.from_numpy(self.dn_orb)[0:390]
            self.x = torch.cat([self.up_orb, self.dn_orb], 0)

            self.y = self.input_orbits.iloc[idx]['Labels']

            # min_val = torch.min(self.x)
            # max_val = torch.max(self.x)
            x_min = -3.56
            x_max = 7.73
            self.x = (self.x - x_min) / (x_max - x_min)

            if self.transform is not None:
                self.x = self.transform(self.x)
            return self.x, self.y

        except IndexError:
            raise IndexError(f"Index {idx} is out of range.")


folder_path = '/storage3/DSIP/Demeter/ASCII_1132'
input_orbits =pd.read_csv('/storage3/DSIP/Demeter/ICE/I_ICE_CleanINPUT_Orbits.csv')
logging.warning('the Orbitdataset class has covered')

dataset = OrbitsDataset(folder_path, input_orbits)
print('len of dataset', len(dataset))
logging.warning('len of dataset is %s', str(len(dataset)))

# print('min_value',min_va)
# %%
train_set, test_set = train_test_split(dataset.input_orbits, test_size=0.34, random_state= 42, shuffle = False)
logging.warning('train-test split done')
val_set, test_set = train_test_split(test_set, test_size = 0.35, random_state =42, shuffle = True)
logging.warning('Val-test split done')

y_train_indices = list(range(len(train_set)))
y_train = [dataset[i][1] for i in y_train_indices]
class_sample_count = np.array([len(np.where(np.array(y_train) == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
samples_weight = np.array([weight[t] for t in y_train])
samples_weight = torch.from_numpy(samples_weight)
sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

train_dataset = OrbitsDataset(folder_path, train_set)
val_dataset = OrbitsDataset(folder_path, val_set)
test_dataset = OrbitsDataset(folder_path, test_set)

logging.warning('train_dataset set')
# %%
print('length of trainset:',len(train_set))
print('length of valset:',len(val_set))
print('length of testset:',len(test_set))
logging.warning('len of trainset is %s, test set is %s, val set is %s', str(len(train_set)),str(len(test_set)),str(len(val_set)))
# %%
logging.warning('approching dataloader')
train_data_loader = DataLoader(train_dataset, batch_size=3)
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=3, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=3)
print('length of train_data_loader:',len(train_data_loader))
print('length of val_data_loader:',len(val_data_loader))
print('length of test_data_loader :',len(test_data_loader))


logging.warning('Dataloading Done')
# %%
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.conv2 = nn.Conv2d(3,6,3)
        self.conv3 = nn.Conv2d(6,9,3)
        self.conv4 = nn.Conv2d(9,12,3)
        self.conv5 = nn.Conv2d(12,15,3)
        self.linear1 = nn.Linear(9900,1000)
        self.linear2 = nn.Linear(1000,500)
        self.linear3 = nn.Linear(500,100)
        self.linear4 = nn.Linear(100,50)
        self.linear5 = nn.Linear(50,1)
        self.maxpool = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x = self.dropout(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = self.sigmoid(x)
        return x


# %%
logging.warning('Model class covered')
# %%
model = net()
loss_fn=nn.BCELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.005)
print('model summary', model)
logging.warning('model summary is %s', str(model))

# %%
num_epoch=60
train_loss=[]
val_loss = []
best_val_loss = 10  # Initialize with a large value
patience = 13 # Number of epochs without improvement to wait before stopping
counter = 0
logging.warning('training started')
for epoch in range(num_epoch):
  train_epoch_loss=0
  val_epoch_loss=0
  model.train()
  for i, (data, target) in enumerate(train_data_loader):
    data =  data.unsqueeze(1)
    
    output=model(data.float())
    loss=loss_fn(output,target.unsqueeze(1).float())
    train_epoch_loss += loss.detach().numpy()#not accumulating the gradient and converting tensor to array and storing
    # print(i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i == 100:
       break 
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
fig = plt.figure(figsize=(10,8))
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='val loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(frameon=False)
plt.title('model_ICE-D1-s [batch size = 3, lr =0.005 ]')
plt.savefig('Test-train_ICE_D1-s.png')
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
plt.legend(frameon=True)
plt.xlabel('Disciminator output')
plt.show()
plt.savefig('Disciminator output_ICE_D1-s.png')

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
plt.savefig('Confusion matrics_ICE_D1-s.png')
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
plt.savefig('ROC_ICE_D1-s.png')




