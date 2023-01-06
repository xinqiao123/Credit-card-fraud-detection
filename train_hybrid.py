
import pandas as pd 
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import math 
import numpy as np 
from torch import optim
from sklearn.metrics import roc_auc_score, f1_score
import sklearn 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler 

parser = argparse.ArgumentParser(description='Training for credit card fraud detection.')
parser.add_argument('--dataset', type=str, default='ieee_cis_transaction')
parser.add_argument('--model', type=str, default='tab_transformer')

# hyperparameters for tabtransformer model configuration
parser.add_argument('--num_blocks', type=int, default=1, help="No of encoder blocks.")  
parser.add_argument('--heads', type=int, default=1, help="No of heads.") 
parser.add_argument('--dims', type=int, default=1, help="Embedding dimemsion.")  
parser.add_argument('--dropout', type=float, default=0.4, help="Dropout rate.")   


# hyperparameters for model training 
parser.add_argument('--batch_size', type=int, default=1024, help="Batch size.") 
parser.add_argument('--learning_rate', type=int, default=0.0001, help="Learning rate.")
parser.add_argument('--weight_decay', type=int, default=0.03, help="Weight decay.")  
parser.add_argument('--runs', type=int, default=1, help="No of runs.")   
parser.add_argument('--epochs', type=int, default=50, help="No of epochs.")  
parser.add_argument('--early_stopping_rounds', type=int, default=30, help="Early stop.") 

args = parser.parse_args()


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
        data = pd.read_csv(path, index_col='TransactionID')
        label = 'isFraud'

        cat_index =  [1, 6, 11, 12, 34, 35, 36, 37, 38, 39, 40, 41, 171, 173, 174, 179, 180, 181, 182, 183, 184, 185, 186, 187]
        num_features = 216
        categories_unique = []
        for i in cat_index:
            col = data.columns[i]
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            categories_unique.append(len(le.classes_))

        num_index = list(set(range(num_features)) - set(cat_index))
        num_continuous = num_features - len(cat_index)
        # 
        # for col in df.columns[cat_index]:
        #     categories_unique.append(len(df[col].unique())) 

        X = data.iloc[:,:-1].to_numpy()
        y = data[label].to_numpy()

    else:
        raise AttributeError("Dataset \"" + dataset + "\" not available")
    return X, y,  num_features, cat_index, num_index, num_continuous, categories_unique




X, y, num_features, cat_index, num_index, num_continuous, categories_unique = load_data(args.dataset)

scaler = StandardScaler()

for n in num_index:
    X[:,[n]] = scaler.fit_transform(X[:,n].reshape(-1, 1))


# train val split according to the time 
X_train, X_val = X[:3*X.shape[0]//4], X[3*X.shape[0]//4:]
y_train, y_val = y[:3*X.shape[0]//4], y[3*X.shape[0]//4:]

print("Train size:", X_train.shape)
print("Val size:", X_val.shape)

import torch 
from torch.utils.data import TensorDataset, DataLoader

X_train, X_val = torch.tensor(X_train).float(), torch.tensor(X_val).float()
y_train, y_val = torch.tensor(y_train).float(), torch.tensor(y_val).float()
  


dim = args.dims
depth = args.num_blocks
heads = args.heads
dropout = args.dropout
batch_size = args.batch_size

device = torch.device('cpu')

from torch import nn 
import torch 



import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=16,
            dropout=0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


# transformer

class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim) # (Embed the categorical features.)
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim, dropout=ff_dropout))),
            ]))

    def forward(self, x):

        # print( torch.min(x), torch.max(x))
        x = self.embeds(x)

        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)

        return x


# mlp

class MLP(nn.Module):
    def __init__(self, dims, act=None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims_pairs) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                self.dim_out = dim_out
                continue

            act = default(act, nn.ReLU())
            layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)

        # Added for multiclass output!
        if self.dim_out > 1:
            x = torch.softmax(x, dim=1)
        return x


# main class

class TabTransformerModel(nn.Module):
    def __init__(
            self,
            *,
            categories,
            num_continuous,
            dim,
            depth,
            heads,
            dim_head=16,
            dim_out=1,
            mlp_hidden_mults=(4, 2),
            mlp_act=None,
            num_special_tokens=2,
            continuous_mean_std=None,
            attn_dropout=0.,
            ff_dropout=0.
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations

        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        total_tokens = self.num_unique_categories + num_special_tokens
        # print(total_tokens)
        # for automatically offsetting unique category ids to the correct position in the categories embedding table

        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value=num_special_tokens) # Prepend num_special_tokens.
        categories_offset = categories_offset.cumsum(dim=-1)[:-1]
        self.register_buffer('categories_offset', categories_offset)

        # continuous

        if exists(continuous_mean_std):
            assert continuous_mean_std.shape == (num_continuous,
                                                 2), f'continuous_mean_std must have a shape of ({num_continuous}, 2) where the last dimension contains the mean and variance respectively'
        self.register_buffer('continuous_mean_std', continuous_mean_std)

        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous

        # transformer

        self.transformer = Transformer(
            num_tokens=total_tokens,
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )

        # mlp to logits

        input_size = (dim * self.num_categories) + num_continuous
        l = input_size // 8

        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]

        self.mlp = MLP(all_dimensions, act=mlp_act)

    def forward(self, x_categ, x_cont):

        # Adaptation to work without categorical data
        if x_categ is not None:
            assert x_categ.shape[-1] == self.num_categories, f'you must pass in {self.num_categories} ' \
                                                             f'values for your categories input'
            x_categ += self.categories_offset
            x = self.transformer(x_categ)
            flat_categ = x.flatten(1)

        assert x_cont.shape[1] == self.num_continuous, f'you must pass in {self.num_continuous} ' \
                                                       f'values for your continuous input'

        if exists(self.continuous_mean_std):
            mean, std = self.continuous_mean_std.unbind(dim=-1)
            x_cont = (x_cont - mean) / std

        normed_cont = self.norm(x_cont)

        # Adaptation to work without categorical data 
        if x_categ is not None:
            x = torch.cat((flat_categ, normed_cont), dim=-1)
        else:
            x = normed_cont

        return x, self.mlp(x)

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                        shuffle=False)



val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                        shuffle=False)    


model = TabTransformerModel(
            categories=categories_unique, 
            num_continuous=num_continuous, 
            dim_out = 1, 
            mlp_act=nn.ReLU(),
            dim = dim, 
            depth = depth,
            heads = heads,
            attn_dropout= dropout, 
            ff_dropout=dropout, 
            mlp_hidden_mults=(2,1))




model.load_state_dict(torch.load('./results/ieee_cis_transaction/TabTransformer/best_checkpoint.pt'))

model.eval()
predictions = []

intermidate = []

with torch.no_grad():
    for batch_X in train_loader:
        x_categ = batch_X[0][:, cat_index].int().to(device) 
        x_cont = batch_X[0][:, num_index].to(device)

        inter, preds = model(x_categ, x_cont)
        # preds = model( x_cont,x_categ)
        preds = torch.sigmoid(preds)
        predictions.append(preds.cpu())
        intermidate.append(inter)

train_features = torch.cat(intermidate, 0)
print(train_features)


predictions = []
intermidate = []
with torch.no_grad():
    for batch_X in val_loader:
        x_categ = batch_X[0][:, cat_index].int().to(device) 
        x_cont = batch_X[0][:, num_index].to(device)

        inter, preds = model(x_categ, x_cont)
        # preds = model( x_cont,x_categ)
        preds = torch.sigmoid(preds)
        predictions.append(preds.cpu())
        intermidate.append(inter)

val_features = torch.cat(intermidate, 0)
print(val_features)

import xgboost as xgb 

AUC = []
for i in range(1):
    clf = xgb.XGBClassifier( 
            n_estimators=2000,
            max_depth=12, 
            learning_rate=0.02,  
            subsample=0.8,
            colsample_bytree=0.4, 
            missing=-1, 
            eval_metric='auc',
            seed = i,
            # USE GPU
            tree_method='gpu_hist' )
    print()
    print('%sth runs of XGBoost:'% (i+1))       
    history = clf.fit(train_features, y_train, 
            eval_set=[(val_features,y_val)],
            verbose=50, early_stopping_rounds=100)

    y_probs = clf.predict_proba(val_features)[:,1]
    
    auc = roc_auc_score(y_val,y_probs)
    AUC.append(auc)
    print ('AUC:', auc)

print('AUC mean ± dev of 10 runs:',np.mean(AUC),'±', np.std(AUC))