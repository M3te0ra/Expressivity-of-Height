import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import torch
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import MulticlassF1Score
from ptflops import get_model_complexity_info

import matplotlib.pyplot as plt


df = pd.read_csv('BankChurners.csv')
df.head()

df.iloc[:,:10].head()

df.iloc[:,:10].describe(include='all')

df.iloc[:,10:].head()

df.iloc[:,10:].describe(include='all')

df = df.rename(columns={df.columns[-2]: 'attrition1', df.columns[-1]: 'attrition2'})
df.head()

df.duplicated().sum()

df_processed = pd.DataFrame()

df['Attrition_Flag'].unique()

df_processed['attrition_flag'] = df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})
df_processed.head()

df_processed['age'] = df['Customer_Age']
df_processed.head()

df_processed['gender'] = df['Gender'].map({'F': 0, 'M': 1})
df_processed.head()

df_processed['dependent_count'] = df['Dependent_count']
df_processed.head()


df['Education_Level'].unique()

education_unknown_indexes = df[df['Education_Level'] == 'Unknown'].index

df = df[~df.index.isin(education_unknown_indexes)]
df_processed = df_processed[~df_processed.index.isin(education_unknown_indexes)]
df_processed.describe(include='all')

education_level_dict = {
    'Uneducated': 0,
    'High School': 1,
    'College': 2,
    'Graduate': 3,
    'Post-Graduate': 4,
    'Doctorate': 5
}

df_processed['education'] = df['Education_Level'].map(education_level_dict)
df_processed.head()

df['Marital_Status'].unique()


unknown_marital_indexes = df[df['Marital_Status'] == 'Unknown'].index

df = df[~df.index.isin(unknown_marital_indexes)]
df_processed = df_processed[~df_processed.index.isin(unknown_marital_indexes)]
df_processed.describe(include='all')

marital_dummies = pd.get_dummies(df['Marital_Status'], prefix='status', drop_first=True)
marital_dummies.head()

df_processed = pd.concat([df_processed, marital_dummies], axis='columns')
df_processed.head()


df['Income_Category'].unique()

unknown_income_indexes = df[df['Income_Category'] == 'Unknown'].index

df = df[~df.index.isin(unknown_income_indexes)]
df_processed = df_processed[~df_processed.index.isin(unknown_income_indexes)]
df_processed.describe(include='all')

income_dict = {
    'Less than $40K': 0,
    '$40K - $60K': 1,
    '$60K - $80K': 2,
    '$80K - $120K': 3,
    '$120K +': 4
}

df_processed['income_category'] = df['Income_Category'].map(income_dict)
df_processed.head()


df['Card_Category'].unique()

card_dummies = pd.get_dummies(df['Card_Category'], prefix='card', drop_first=True)
card_dummies.head()

df_processed = pd.concat([df_processed, card_dummies], axis='columns')
df_processed.head()

rest_of_data = df.iloc[:,9:21]
rest_of_data.head()

df_processed = pd.concat([df_processed, rest_of_data], axis='columns')
df_processed.head()

from sklearn.utils.class_weight import compute_class_weight

y = df_processed['attrition_flag']
y.value_counts()

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights

class_weight_dict = {
    0: class_weights[0],
    1: class_weights[1]
}
class_weight_dict

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(df_processed.drop('attrition_flag', axis='columns'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2022)


y_train = pd.Series.to_numpy(y_train)
y_test = pd.Series.to_numpy(y_test)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, input_dim, size):
        super(Model, self).__init__()
        self.layer11 = nn.Linear(input_dim, width//2)
        self.layer12 = nn.Linear(input_dim, width//2)
        self.layer21 = nn.Linear(size, width//2)
        self.layer22 = nn.Linear(size, width//2)
        self.layer3 = nn.Linear(size, 2)

    def forward(self, x):
        l11 = F.relu(self.layer11(x))
        l12 = F.relu(self.layer12(x)+l11)

        l21 = F.relu(self.layer21(torch.cat((l11,l12),1)))
        l22 = F.relu(self.layer22(torch.cat((l11,l12),1))+l21)

        x = F.softmax(self.layer3(torch.cat((l21,l22),1)), dim=1)
        return x




width = 96
model     = Model(X_train.shape[1], width)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_fn   = nn.CrossEntropyLoss(weight=Variable(torch.from_numpy(class_weights)).float())

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)




metric_mircoF1 = MulticlassF1Score(num_classes=2, average='macro')
metric_marcoF1 = MulticlassF1Score(num_classes=2, average='macro')


import tqdm

EPOCHS  = 2000

X_train = Variable(torch.from_numpy(X_train)).float()
y_train = Variable(torch.from_numpy(y_train)).long()
X_test  = Variable(torch.from_numpy(X_test)).float()
y_test  = Variable(torch.from_numpy(y_test)).long()

loss_list     = np.zeros((EPOCHS,))
microF1_list = np.zeros((EPOCHS,))
macroF1_list = np.zeros((EPOCHS,))

import time

time_start = time.time()

for epoch in tqdm.trange(EPOCHS):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss_list[epoch] = loss.item()
    
    # Zero gradients
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        y_pred = model(X_test)
        #correct = (torch.argmax(y_pred, dim=1) == y_test).type(torch.FloatTensor)
        mircoF1= metric_mircoF1(torch.argmax(y_pred, dim=1), y_test)
        marcoF1= metric_marcoF1(torch.argmax(y_pred, dim=1), y_test)
        microF1_list[epoch] = mircoF1.mean()
        macroF1_list[epoch] = marcoF1.mean()

time_end = time.time()

print((time_end-time_start)/10000)

print(max(microF1_list))
print(max(macroF1_list))















