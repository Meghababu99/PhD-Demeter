# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, matthews_corrcoef
import seaborn as sns
import optuna

#The given code is an example for LSTM classifier. Huperparameter tuning is done using Optuna

# %%

# Seed for reproducibility
torch.manual_seed(21894)
np.random.seed(21894)


# %%

class TimeSeriesDataset(Dataset):
    def __init__(self, df, min_data_points=34, transform=None):
        self.transform = transform
        self.min_data_points = min_data_points
        self.df = df
        self.features = self.df.filter(regex='^Res_fb_').columns
        self.labels = self.df['label'].values
        self.sequences, self.label_sequences = self.create_half_orbit_sequences(self.df)

    def create_half_orbit_sequences(self, df):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DateTimeIndex.")
        
        df = df.sort_index()
        sequences, label_sequences = [], []
        features = df.filter(regex='^Res_fb_').columns
        label_column = 'label'
        gap_threshold = pd.Timedelta(hours=1)
        gaps = df.index.to_series().diff() > gap_threshold
        gap_indices = gaps[gaps].index
        segment_boundaries = [df.index[0]] + gap_indices.tolist() + [df.index[-1]]
        two_hours = pd.Timedelta(hours=2)
        for i in range(0, len(segment_boundaries) - 2, 2):
            start, mid, end = segment_boundaries[i], segment_boundaries[i + 1], segment_boundaries[i + 2]

            # Skip sequence creation if time difference between segments is >= 2 hours
            if mid - start >= two_hours or end - mid >= two_hours:
                # print(f"Skipping sequence between: {start} and {end} due to time gap >= 2 hours")
                continue 
            seq1, seq2 = df.loc[start:mid, features], df.loc[mid:end, features]
            lab1, lab2 = df.loc[start:mid, label_column], df.loc[mid:end, label_column]
            combined_seq, combined_lab = pd.concat([seq1, seq2]), pd.concat([lab1, lab2])
            
            if len(combined_seq) >= self.min_data_points:
                sequences.append(combined_seq[:self.min_data_points].values)
                label_sequences.append(combined_lab[:self.min_data_points].values)
        
        return sequences, label_sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sample, label = self.sequences[idx], self.label_sequences[idx]
        y = 1 if sum(label) >= 1 else 0
        x, y = torch.tensor(sample, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


# %%
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, num_layers, output_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_size1 = hidden_size1
        # self.hidden_size2 = hidden_size2
        # self.hidden_size3 = hidden_size3
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size1, output_size)
        # self.fc2 = nn.Linear(hidden_size2, output_size)
        # self.fc3 = nn.Linear(hidden_size3, output_size)# in the first trial the 3 fc layers were used 
        self.relu = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size1).to(x.device)
        out, (h, _) = self.lstm(x, (h0, c0))
        x = self.fc1(out[:, -1, :])
        # x = self.relu(x)
        # x = self.fc2(x)
        # x = self.relu(x)
        # x = self.fc3(x)
        return x

# %%

def load_data():
    data = pd.read_pickle('/storage3/DSIP/Demeter/Newdataset/Down_Orbits-max-val-location-2T20W.pkl')
    df = data.loc[:, ~data.columns.str.startswith('Q3')]
    eq = pd.read_csv("/home/mbabu/EQ_Data/EQ.csv", parse_dates=['Time'])
    eq['Time'] = pd.to_datetime(eq['Time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    eq['Time'] = pd.to_datetime(eq['Time'])
    return df, eq

def preprocess_data(df, seq=34):
    train_set, test1_set = train_test_split(df, test_size=0.413839, random_state=42, shuffle=False)
    test_set = pd.DataFrame(test1_set[test1_set.index < '2009-01-01'])
    val_set = pd.DataFrame(test1_set[test1_set.index > '2009-01-01'])
    train_dataset = TimeSeriesDataset(train_set, seq)
    val_dataset = TimeSeriesDataset(val_set, seq)
    test_dataset = TimeSeriesDataset(test_set, seq)
    return train_dataset, val_dataset, test_dataset

def scale_data(train_dataset, val_dataset, test_dataset):
    scaler = StandardScaler()
    train_data = [data for data, _ in train_dataset]
    train_data = torch.cat(train_data).numpy()
    scaler.fit(train_data.reshape(-1, train_data.shape[-1]))
    
    def scale(dataset):
        scaled_data = []
        for data, target in dataset:
            x = data.numpy()
            x = scaler.transform(x)
            x = x.reshape(data.shape)
            scaled_data.append((torch.from_numpy(x), target))
        return scaled_data
    
    return scale(train_dataset), scale(val_dataset), scale(test_dataset), scaler

def create_dataloaders(scaled_train_data, scaled_val_data, scaled_test_data, batch_size):
    train_loader = DataLoader(scaled_train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(scaled_val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(scaled_test_data, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# %%


def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs, patience):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print(f'Early stopping at epoch {epoch + 1}')
            break
        
        model.train()
        train_loss, correct_train, total_train = 0, 0, 0
        
        for sequences, target in train_loader:
            sequences, target = sequences, target.float()
            optimizer.zero_grad()
            outputs = model(sequences).squeeze()
            if outputs.dim() != 1:  # Example: Output shape is [batch_size]
                continue
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            predicted = (outputs >= 0.5).int()
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(100 * correct_train / total_train)
        
        model.eval()
        val_loss, correct_val, total_val = 0, 0, 0

        with torch.no_grad():
            for sequences, target in val_loader:
                sequences, target = sequences, target.float()
                outputs = model(sequences).squeeze()
                loss = criterion(outputs, target)
                val_loss += loss.item()
                predicted = (outputs >= 0.5).int()
                total_val += target.size(0)
                correct_val += (predicted == target).sum().item()
        
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(100 * correct_val / total_val)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Val Acc: {val_accuracies[-1]:.2f}%')

        # Check for early stopping
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            early_stop = True

    return train_losses, val_losses, train_accuracies, val_accuracies


# %%

def evaluate_model(model, test_loader, criterion):
    model.eval()
    y_true, y_pred, reconstruction_errors,binary_predictions = [], [], [],[]
    
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
    # data = data.unsqueeze(1)
            sequences, target = data, target.float()
            outputs = model(sequences).squeeze()
            loss = criterion(outputs, target)
            output=model(data)
            reconstruction_errors.append(loss.item())
            y_pred.append(output.detach().numpy())
            y_true.append(target.detach().numpy())
        y_pred = np.concatenate(y_pred).squeeze()
        y_true = np.concatenate(y_true)
        binary_predictions = (y_pred >= 0.5).astype(int)
            
  
    conf_matrix = confusion_matrix(y_true, binary_predictions)
    accuracy = accuracy_score(y_true, binary_predictions)
    fpr, tpr, thresholds = roc_curve(y_true, binary_predictions)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, binary_predictions)
    f1 = f1_score(y_true, binary_predictions)
    mcc = matthews_corrcoef(y_true, binary_predictions)
    
    return accuracy, conf_matrix, roc_auc, precision, recall, f1, mcc, reconstruction_errors,y_true, y_pred,binary_predictions,fpr,tpr


# %%

def objective(trial):
    input_size = len(train_dataset.features)
    output_size = 1
    hidden_size1 = trial.suggest_int('hidden_size1', 6, 34)
    # hidden_size2 = trial.suggest_int('hidden_size2', 6, 24)
    # hidden_size3 = trial.suggest_int('hidden_size3', 6, 24)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-3)
    
    model = LSTMClassifier(input_size, hidden_size1,  num_layers, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_epochs = 100
    
    train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs,patience = 5)
    accuracy, conf_matrix, roc_auc, precision, recall, f1, mcc, reconstruction_errors,y_true, y_pred,binary_predictions,fpr,tpr = evaluate_model(model, test_loader, criterion)
    
    return f1


# %%

if __name__ == "__main__":
    df, eq = load_data()
    seq = 34
    batch_size =8
    patience = 5
    train_dataset, val_dataset, test_dataset = preprocess_data(df, seq)
    scaled_train_data, scaled_val_data, scaled_test_data, scaler = scale_data(train_dataset, val_dataset, test_dataset)
    train_loader, val_loader, test_loader = create_dataloaders(scaled_train_data, scaled_val_data, scaled_test_data,batch_size)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=150)
    
    print("Best hyperparameters:", study.best_params)
    print("Best F1 score:", study.best_value)



    


# %%
input_size = len(train_dataset.features)
output_size = 1
best_model = LSTMClassifier(input_size, study.best_params['hidden_size1'],study.best_params['num_layers'], output_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(best_model.parameters(), lr=study.best_params['learning_rate'], weight_decay=study.best_params['weight_decay'])
hs1=study.best_params['hidden_size1']
nl=study.best_params['num_layers']
lr =study.best_params['learning_rate']
# %%
print('for the best model' )
num_epochs = 100
train_losses, val_losses, train_accuracies, val_accuracies = train_model(best_model, criterion, optimizer, train_loader, val_loader, num_epochs,patience = 5)

# %%
accuracy, conf_matrix, roc_auc, precision, recall, f1, mcc, reconstruction_errors,y_true, y_pred,binary_prediction,fpr,tpr = evaluate_model(best_model, test_loader, criterion)

print(f"Test Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"ROC AUC: {roc_auc}")
print(f"F1 Score: {f1}")
print(f"MCC: {mcc}")

plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), val_losses, label='Validation Loss')
details = (
    f'input_size: {input_size}\n'
    f'hidden_dim: {[hs1]}\n'
    f'num_layers: {nl}\n'
    f'batch_size: {batch_size}\n'
    f'Seq_length: {seq} data points\n'
    f'learning_rate: {lr}'
)
plt.text(0.80, 0.95, details, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', horizontalalignment='right',
         bbox=dict(facecolor='white', alpha=0.5))
plt.title('Train and val Loss Curves')
plt.legend()
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/Loss_curve-3.png')
plt.show()


plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy Curve')
plt.legend()
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/Acc-curve-3.png')
plt.show()


# %%
plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix Test data model')
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/Confusion-Matrix-3.png')
plt.show()



plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/ROC-3.png')
plt.show()

# %%

plt.figure(figsize=(10, 6))
plt.hist(reconstruction_errors, bins=50, edgecolor='black')
plt.xlabel('Reconstruction Error')
plt.ylabel('Frequency')
plt.title('Distribution of Reconstruction Errors for test_data')
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/Error-Test-3.png')
plt.grid(True)
# plt.savefig(f'C:\PROJECT-DEMETER\Anomaly\LSTM\AE-FULL_FEATURES-2orbits\M3-ReError.png')


# %%
plt.figure()
plt.hist(binary_prediction[y_true==0], bins=200, alpha=0.5, label='Non_seismic')
plt.hist(binary_prediction[y_true==1], bins=200, alpha=0.5, label='seismic')
plt.legend()
plt.title('Classification')
plt.xlabel('Disciminator output')
plt.ylabel('Counts')
plt.savefig('/home/mbabu/GRID-METHODS/LSTM-Classifier/Results/Disciminator-Test-3.png')
plt.show()
# plt.savefig('Disciminator output_ICE-C1-s.png')


# %%


# %%



