import gc
import pickle as pkl

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.feature import get_combine

class FeedForwardNN(nn.Module):

    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes,
               output_size, emb_dropout, lin_layer_dropouts):

        """
        Parameters
        ----------

        emb_dims: List of two element tuples
          This list will contain a two element tuple for each
          categorical feature. The first element of a tuple will
          denote the number of unique values of the categorical
          feature. The second element will denote the embedding
          dimension to be used for that feature.

        no_of_cont: Integer
          The number of continuous features in the data.

        lin_layer_sizes: List of integers.
          The size of each linear layer. The length will be equal
          to the total number
          of linear layers in the network.

        output_size: Integer
          The size of the final output.

        emb_dropout: Float
          The dropout to be used after the embedding layers.

        lin_layer_dropouts: List of floats
          The dropouts to be used after each linear layer.
        """

        super().__init__()

        # Embedding layers
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                         for x, y in emb_dims])

        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.no_of_cont = no_of_cont

        # Linear Layers
        first_lin_layer = nn.Linear(self.no_of_embs + self.no_of_cont, lin_layer_sizes[0])

        self.lin_layers =\
            nn.ModuleList([first_lin_layer] +\
                [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i + 1])
                    for i in range(len(lin_layer_sizes) - 1)])

        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        # Output Layer
        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        # Batch Norm Layers
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size)
                                        for size in lin_layer_sizes])

        # Dropout Layers
        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.droput_layers = nn.ModuleList([nn.Dropout(size)
                                      for size in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):

        if self.no_of_embs != 0:
            x = [emb_layer(cat_data[:, i])
                for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
#             normalized_cont_data = self.first_bn_layer(cont_data)

            if self.no_of_embs != 0:
                x = torch.cat([x, cont_data], 1) 
            else:
                x = cont_data

        for lin_layer, dropout_layer, bn_layer in\
            zip(self.lin_layers, self.droput_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)
        x = self.output_layer(x)

        return x
    
    
class TabularDataset(Dataset):
    def __init__(self, data, cat_cols=None, output_col=None):
        """
        Characterizes a Dataset for PyTorch

        Parameters
        ----------

        data: pandas data frame
          The data frame object for the input data. It must
          contain all the continuous, categorical and the
          output columns to be used.

        cat_cols: List of strings
          The names of the categorical columns in the data.
          These columns will be passed through the embedding
          layers in the model. These columns must be
          label encoded beforehand. 

        output_col: string
          The name of the output variable column in the data
          provided.
        """
        self.n = data.shape[0]

        if output_col:
            self.y = data[output_col].astype(np.float32).values.reshape(-1, 1)
        else:
            self.y =  np.zeros((self.n, 1))

        self.cat_cols = cat_cols if cat_cols else []
        self.cont_cols = [col for col in data.columns
                          if col not in self.cat_cols + [output_col]]

        if self.cont_cols:
            self.cont_X = data[self.cont_cols].astype(np.float32).values
        else:
            self.cont_X = np.zeros((self.n, 1))

        if self.cat_cols:
            self.cat_X = data[cat_cols].astype(np.int64).values
        else:
            self.cat_X =  np.zeros((self.n, 1))

    def __len__(self):
        """
        Denotes the total number of samples.
        """
        return self.n

    def __getitem__(self, idx):
        """
        Generates one sample of data.
        """
        return [self.y[idx], self.cont_X[idx], self.cat_X[idx]]

    
def embed_target(target_col, target_dim=5, batch_size=2048):
    def get_device():
        # Use GPU if available, otherwise stick with cpu
        use_cuda = torch.cuda.is_available()
        torch.manual_seed(123)
        device = torch.device("cuda" if use_cuda else "cpu")
        return device
    
    not_train = ['txkey', 'date', 'time', 'fraud_ind']
    need_encode = ['acquirer', 'bank', 'card', 'coin', 'mcc', 'shop', 'city', 'nation']
    cat = ['trade_cat', 'pay_type', 'status']
    
    if target_col == 'money':
        loss_fn = nn.MSELoss()
        output_num = 1
    elif target_col == 'trade_type':
        loss_fn = nn.CrossEntropyLoss()
        output_num = 11
    elif target_col == 'online':
        loss_fn = nn.CrossEntropyLoss()
        output_num = 2
    
    combine = get_combine()
    combine = pd.get_dummies(combine, columns=cat + ['fallback', '3ds'])
    
    label_encoders = {}
    for cat_col in need_encode:
        label_encoders[cat_col] = LabelEncoder()
        combine[cat_col] = label_encoders[cat_col].fit_transform(combine[cat_col])
    del label_encoders
    gc.collect()
    
    device = get_device()
    tr_dataset = TabularDataset(
        data=combine.loc[:, [x for x in combine.columns if x not in not_train]], 
        cat_cols=need_encode, 
        output_col=target_col)
    tr_dataloader = DataLoader(tr_dataset, batch_size, shuffle=False, num_workers=1)
    cat_dims = [int(combine[col].nunique()) for col in need_encode]
    emb_dims = [(x, target_dim) for x in cat_dims]
    model = FeedForwardNN(emb_dims, no_of_cont=33, lin_layer_sizes=[33 + target_dim * 8, 33 + target_dim * 8],
                      output_size=output_num, emb_dropout=0.3,
                      lin_layer_dropouts=[0.3, 0.3]).to(device)
    
    print('start training...')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    training_epochs = 50
    for epoch in range(training_epochs):
        model.train()
        for step, (batch_y, batch_cont_x, batch_cat_x) in enumerate(tr_dataloader):
            y_pred = model(batch_cont_x.to(device), batch_cat_x.to(device))
            
            if(target_col == 'money'):
                batch_y = batch_y.to(device)
                loss = loss_fn(y_pred, batch_y)
            else:
                batch_y = batch_y.to(dtype=torch.int64).to(device)
                loss = loss_fn(y_pred, batch_y.squeeze())

            # reset gradients
            optimizer.zero_grad()

            # backward pass
            loss.backward()

            # update weights
            optimizer.step()

            del batch_y, batch_cont_x, batch_cat_x  
        print(f'epoch: {epoch}, loss: {loss}')
    df = pd.DataFrame()

    for i, col in enumerate(need_encode):
        temp = pd.DataFrame(
            data = Variable(
                model.emb_layers[i](torch.from_numpy(combine[col].values.reshape(-1, 1)).to(device)).reshape(-1, target_dim).cpu()
                ).data.numpy(),
            columns = [f'{col}_embed_{target_col}_{x + 1}' for x in range(target_dim)]
        )
        df = pd.concat([temp, df], axis=1)
    
    return df