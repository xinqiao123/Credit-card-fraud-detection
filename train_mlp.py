import pandas as pd 
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import math 
import numpy as np 
from torch import optim
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler 

parser = argparse.ArgumentParser(description='Training for credit card fraud detection.')
parser.add_argument('--dataset', type=str, default='synthetic_transaction')
parser.add_argument('--model', type=str, default='ResNet')

# hyperparameters for model configuration 
parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate.")   


# hyperparameters for model training 
parser.add_argument('--batch_size', type=int, default=1024, help="Batch size.") 
parser.add_argument('--learning_rate', type=int, default=0.0001, help="Learning rate.")
parser.add_argument('--weight_decay', type=int, default=0.03, help="Weight decay.")  
parser.add_argument('--runs', type=int, default=10, help="No of runs.")   
parser.add_argument('--epochs', type=int, default=200, help="No of epochs.")  
parser.add_argument('--early_stopping_rounds', type=int, default=30, help="Early stop.") 

args = parser.parse_args()



import time

class Timer:
    def __init__(self):
        self.save_times = []
        self.start_time = 0

    def start(self):
        self.start_time = time.process_time()

    def end(self):
        end_time = time.process_time()
        self.save_times.append(end_time - self.start_time)

    def get_average_time(self):
        return np.mean(self.save_times)



def fraudEncoder(X):
    fraud = (X == 'Yes').astype(int)
    return pd.DataFrame(fraud)

def nanNone(X):
    return X.where(pd.notnull(X), 'None')

def nanZero(X):
    return X.where(pd.notnull(X), 0)

def amountEncoder(X):
    amt = X.apply(lambda x: x[1:]).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
    return pd.DataFrame(amt)


def amountEncoder1(X):
    amt = X.apply(lambda x: x).astype(float).apply(lambda amt: max(1, amt)).apply(math.log)
    return pd.DataFrame(amt)

def timeEncoder(X):
    X_hm = X['Time'].str.split(':', expand=True)
    d = pd.to_datetime(dict(year=X['Year'], month=X['Month'], day=X['Day'], hour=X_hm[0], minute=X_hm[1])).values.astype(
        int)
    return pd.DataFrame(d)



def load_data(dataset):
    print("Loading dataset " + dataset)

    if dataset == 'synthetic_transaction':
        path = './data/card_transaction.v2.csv'
        data = pd.read_csv(path, index_col=0)
        data['Is Fraud?'] = fraudEncoder(data['Is Fraud?'])

        # missing values
        data['Errors?'] = nanNone(data['Errors?'])
        data['Zip'] = nanZero(data['Zip'])
        data['Merchant State'] = nanNone(data['Merchant State'])
        data['Use Chip'] = nanNone(data['Use Chip'])
        data['Amount'] = amountEncoder(data['Amount'])

        timestamp = timeEncoder(data[['Year', 'Month', 'Day', 'Time']])
        scaler = MinMaxScaler()
        data['Timestamp'] = scaler.fit_transform(timestamp)

        # label encoder and calculate the unique number for categorical features 
        cat_features = ['User', 'Card', 'Use Chip', 'Merchant Name',
                        'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']
        categories_unique = []

        for col in cat_features:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            categories_unique.append(len(le.classes_))

        
        label = 'Is Fraud?'
        features = ['User', 'Card', 'Timestamp', 'Amount', 'Use Chip', 'Merchant Name',
                    'Merchant City', 'Merchant State', 'Zip', 'MCC', 'Errors?']


        cat_index = [0,1,4,5,6,7,8,9,10]
        num_features = 11

        num_index = list(set(range(num_features)) - set(cat_index))
        num_continuous = num_features - len(cat_index)

        X = data[features].to_numpy()
        y = data[label].to_numpy()        

    elif dataset == 'ieee_cis_transaction':
        path = './data/ieee_cis_transaction_data.csv'
        df = pd.read_csv(path, index_col='TransactionID')
        label = 'isFraud'

        cat_index =  [1, 6, 11, 12, 34, 35, 36, 37, 38, 39, 40, 41, 171, 173, 174, 179, 180, 181, 182, 183, 184, 185, 186, 187]
        num_features = 216

        num_index = list(set(range(num_features)) - set(cat_index))
        num_continuous = num_features - len(cat_index)
        categories_unique = []
        for col in df.columns[cat_index]:
            categories_unique.append(len(df[col].unique())) 

        X = df.iloc[:,:-1].to_numpy()
        y = df[label].to_numpy()

    else:
        raise AttributeError("Dataset \"" + dataset + "\" not available")
    return X, y,  num_features, cat_index, num_index, num_continuous, categories_unique




X, y,  num_features, cat_index, num_index, num_continuous, categories_unique = load_data(args.dataset)
print('Categorical unique values:', categories_unique)

# the loss has 0.01 improvement with this scaler,
scaler = StandardScaler()
for n in num_index:
    X[:,[n]] = scaler.fit_transform(X[:,n].reshape(-1, 1))

# train val split according to the time 
X_train, X_val = X[:3*X.shape[0]//4], X[3*X.shape[0]//4:]
y_train, y_val = y[:3*X.shape[0]//4], y[3*X.shape[0]//4:]

print("Train size:", X_train.shape)
print("Val size:", X_val.shape)



dropout = args.dropout
batch_size = args.batch_size


from torch import nn 
import torch 
from torch.utils.data import TensorDataset, DataLoader
from models.tabtransformer import TabTransformerModel
from models.mlp import MLP_Model

from torch.utils.data import Dataset, DataLoader

class TabularDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data.astype('float')
        self.y_data = y_data.astype('float')
    def __len__(self):
        return len(self.X_data)
    def __getitem__(self,i):
        return self.X_data[i], self.y_data[i]

train_dataset = TabularDataset(X_train, y_train)
val_dataset = TabularDataset(X_val, y_val)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# device = torch.device('cpu')



AUC = []
F1 = []

class MLP (nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(nn.Linear(input_dim, 8),
                                   nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(8, 4),
                                   nn.ReLU())
        # self.layer3 = nn.Sequential(nn.Linear(64, 32),
        #                            nn.ReLU())
        # self.layer4 = nn.Sequential(nn.Linear(32, 16),
        #                            nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(4, 1),
                                   nn.Sigmoid())
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.layer1(x))
        x = self.dropout(self.layer2(x))
        # x = self.dropout(self.layer3(x))
        # x = self.dropout(self.layer4(x))
        x = self.layer5(x)
        return x 



from rtdl import ResNet





for run in range(args.runs):

    # train_time = Timer()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'mlp':
        model = MLP(input_dim=num_features).to(device)
    elif args.model == 'ResNet':    
        model = ResNet.make_baseline(
            d_in=num_features,
            n_blocks=4,
            d_main=128,
            d_hidden=32,
            dropout_first=0.25,
            dropout_second=0.0,
            d_out=1).to(device)
        
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loss_func = nn.BCEWithLogitsLoss()

    # X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
    # y_train, y_val = torch.tensor(y_train).float(), torch.tensor(y_val).float()

    # train_dataset = TensorDataset(X_train, y_train)
    # train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
    #                         shuffle=False)

    # val_dataset = TensorDataset(X_val, y_val)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
    #                         shuffle=False)      


    min_val_loss = float("inf")  # a big value 
    min_val_loss_idx = 0                        

    loss_history = []
    val_loss_history = []


    # train_time.start()
    for epoch in range(args.epochs):
        for i, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_X.float())

            out = out.squeeze()

            loss = loss_func(out, batch_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        loss_history.append(loss.item())

        # Early Stopping
        val_loss = 0.0
        val_dim = 0
        for val_i, (batch_val_X, batch_val_y) in enumerate(val_loader):
            batch_val_X = batch_val_X.to(device)
            batch_val_y = batch_val_y.to(device)

            out = model(batch_val_X.float())
            out = out.squeeze()

            val_loss += loss_func(out, batch_val_y.float())
            val_dim += 1
        val_loss /= val_dim
        val_loss_history.append(val_loss.item())

        print("Epoch %d: Train Loss %.5f, Val Loss %.5f" % (epoch, loss.item(), val_loss))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            min_val_loss_idx = epoch

            # Save the currently best model
            torch.save(model.state_dict(), 'best_checkpoint.pt')

        if min_val_loss_idx + args.early_stopping_rounds < epoch:
            print("Validation loss has not improved for %d steps!" % args.early_stopping_rounds)
            print("Early stopping applies.")
            break


    # train_time.end()

    model.load_state_dict(torch.load('best_checkpoint.pt'))


    model.eval()
    predictions = []

    with torch.no_grad():
        for batch_X, _ in val_loader:
            batch_X = batch_X.to(device) 
            preds = model(batch_X.float())
            preds = torch.sigmoid(preds)
            predictions.append(preds.cpu())

    probabilies = np.concatenate(predictions)

    import sklearn 
    import matplotlib.pyplot as plt 

    def plot_roc(name, labels, predictions):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
        print(sklearn.metrics.auc(fp, tp))
        plt.figure()
        plt.plot(100*fp, 100*tp, label=name, linewidth=2)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        # plt.xlim([-0.5,20])
        # plt.ylim([80,100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.show()
    
    def plot_prc(name, labels, predictions,):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
        print(sklearn.metrics.auc(recall, precision))
        plt.figure()
        plt.plot(recall, precision, label=name, linewidth=2)
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        # plt.show()
    

    plot_roc(args.model, y_val, probabilies)
    plot_prc(args.model, y_val, probabilies)

    probabilies = np.concatenate((1 - probabilies, probabilies), 1)

    predictions = np.argmax(probabilies, axis=1)



    auc = roc_auc_score(y_val, probabilies[:,1])
    f1 = f1_score(y_val, predictions)

    print('Runs:',run, ', auc:', auc, ', f1 score:', f1, 
    # ', training time:' , train_time.get_average_time()
    )

    AUC.append(auc)
    F1.append(f1)

AUC_mean = np.mean(AUC)
AUC_std = np.std(AUC)

F1_mean = np.mean(F1)
F1_std = np.std(F1)

print('AUC:', AUC_mean,' ± ', AUC_std)
print('F1:', F1_mean,' ± ', F1_std)




















