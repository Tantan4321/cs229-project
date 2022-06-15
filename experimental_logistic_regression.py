#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
from tqdm import tqdm
import numpy as np


# In[ ]:


assert torch.cuda.is_available(), 'CUDA is not correctly installed!!'
print(torch.tensor([2.0, 3.0, 1.0]).cuda())
property = torch.cuda.get_device_properties(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[ ]:


print('====== Your GPU info ======')
print('name:\t\t', property.name)
print('capability:\t', 'v{}.{}'.format(property.major, property.minor))
print('memory:\t\t', round(property.total_memory / 1e9), 'Gb')
print('processors:\t', property.multi_processor_count)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# LOAD AND SPLIT DATASET HERE

# Load headers
csv_path = "dataset_v2.csv"

with open(csv_path, 'r', newline='') as csv_fh:
    headers = csv_fh.readline().strip().split(',')
    
    
label_col = "Price"
date_col = "Date(UTC)"
# Load features and labels
x_cols = [i for i in range(len(headers)) if (headers[i] != label_col and headers[i] != date_col)]
l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
prices = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

inputs = np.delete(inputs, -1, axis=0)
print(inputs.shape)

# add intercept
new_x = np.zeros((inputs.shape[0], inputs.shape[1] + 1), dtype=inputs.dtype)
new_x[:, 0] = 1
new_x[:, 1:] = inputs

inputs = new_x
print(inputs.shape)

diff = np.diff(prices)
labels = np.where(diff > 0, 1, 0)
print(labels.shape)
print(labels)

# Feature engineering
norm = MinMaxScaler().fit(inputs)
inputs = norm.transform(inputs)

# apply standardization on numerical features
for i in range(inputs.shape[1]):
    
    # fit on training data column
    scale = StandardScaler().fit(inputs[[i]])
    
    # transform the training data column
    inputs[i] = scale.transform(inputs[[i]])


X_train, X_test, y_train,  y_test = train_test_split(
    inputs, labels, test_size=0.1, random_state=42)


# In[ ]:


import torch.nn.functional as F

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.layer1 = torch.nn.Linear(input_dim,50)
        self.layer2 = torch.nn.Linear(50, 10)
        self.layer3 = torch.nn.Linear(10, output_dim)
        
    def forward(self, x):
        x = torch.sigmoid(self.layer1(x))
        x = torch.sigmoid(self.layer2(x))
        x = torch.sigmoid(self.layer3(x)) # To check with the loss function
        return x


# In[6]:


epochs = 1000000
input_dim = inputs.shape[1] # features 
output_dim = 1 # price
learning_rate = 0.005

model = LogisticRegression(input_dim,output_dim).to(device)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# In[7]:


# Convert data to tensors, move onto GPU

X_train, X_test = torch.Tensor(X_train).to(device), torch.Tensor(X_test).to(device)
y_train, y_test = torch.Tensor(y_train).to(device), torch.Tensor(y_test).to(device)


# In[ ]:


# ----------------- TRAIN ------------------
losses = []
losses_test = []
acc = []
acc_test =[]
Iterations = []
iter = 0
step = 10000
for epoch in tqdm(range(int(epochs)),desc='Training Epochs'):
    x = X_train
    labels = y_train
    optimizer.zero_grad() # Setting our stored gradients equal to zero
    outputs = model(X_train)
    loss = criterion(torch.squeeze(outputs), labels) 
    
    loss.backward() # Computes the gradient of the given tensor w.r.t. the weights/bias
    
    optimizer.step() # Updates weights and biases with the optimizer (SGD)
    
    # Print out benchmarking
    if iter%step==0:
        with torch.no_grad():
            # Calculating the loss and accuracy for the test dataset
            correct_test = 0
            total_test = 0
            outputs_test = torch.squeeze(model(X_test))
            loss_test = criterion(outputs_test, y_test)
            
            predicted_test = outputs_test.cpu().round().detach().numpy()
            total_test += y_test.size(0)
            correct_test += np.sum(predicted_test == y_test.cpu().detach().numpy())
            accuracy_test = 100 * correct_test/total_test
            acc_test.append(accuracy_test)
            losses_test.append(loss_test.item())
            
            # Calculating the loss and accuracy for the train dataset
            total = 0
            correct = 0
            total += y_train.size(0)
            correct += np.sum(torch.squeeze(outputs).cpu().round().detach().numpy() == y_train.cpu().detach().numpy())
            accuracy = 100 * correct/total
            acc.append(accuracy)
            losses.append(loss.item())
            Iterations.append(iter)
            
            print(f"Iteration: {iter}. \nTest - Loss: {loss_test.item()}. Accuracy: {accuracy_test}")
            print(f"Train -  Loss: {loss.item()}. Accuracy: {accuracy}\n")
        
    iter+=1


# In[ ]:


# ------------------- PLOT RESULTS -------------------
fig, (ax1, ax2) = plt.subplots(2, 1)

t = np.arange(epochs, step=step)

ax1.plot(t, losses,'r', label='train')
ax1.plot(t, losses_test, 'b', label='test')
ax1.set_xlabel('epochs')
ax1.set_ylabel('loss')
ax1.legend()

ax2.plot(t, acc,'r', label='train')
ax2.plot(t, acc_test, 'b', label='test')
ax2.set_xlabel('epochs')
ax2.set_ylabel('accuracy')
ax2.legend()


# In[ ]:




