#!/usr/bin/env python
# coding: utf-8

# In[6]:


pip install scikit-learn


# In[9]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


# In[2]:


pip install torch torchvision torchaudio


# In[23]:


pip install torch torchvision torchaudio -f https://download.pytorch.org/whl/cpu/torch_stable.html


# In[3]:


import torch
import torch.nn as nn


# In[6]:


import pandas as pd

file_path = r"C:\Users\ASUS\Downloads\netflix.csv"
df = pd.read_csv(file_path)

# Now you can work with the DataFrame
closed_prices = df["Close"]


# In[7]:


seq_len = 15


# In[9]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler
import matplotlib.pyplot as plt

# Read the CSV file and extract the "Close" prices
file_path = r"C:\Users\ASUS\Downloads\netflix.csv"
df = pd.read_csv(file_path)
closed_prices = df["Close"]

# Create a MinMaxScaler instance
mm = MinMaxScaler()

# Transform the data using MinMaxScaler
scaled_price = mm.fit_transform(np.array(closed_prices)[..., None]).squeeze()

# Rest of your code...


# In[10]:


X=[]
Y=[]


# In[11]:


for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i+ seq_len])
    Y.append(scaled_price[i+seq_len])


# In[12]:


X = np.array(X)[... , None]
Y = np.array(Y)[... ,None]


# In[13]:


train_x = torch.from_numpy(X[:int(0.8*X.shape[0])]).float()
train_y = torch.from_numpy(Y[:int(0.8*Y.shape[0])]).float()
test_x = torch.from_numpy(X[:int(0.8*X.shape[0])]).float()
test_y = torch.from_numpy(Y[:int(0.8*Y.shape[0])]).float()


# In[14]:


class Model(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size,batch_first=True)
        self.fc=nn.Linear(hidden_size, 1)
    def forward(self , x):
        output,(hidden,cell)=self.lstm(x)
        return self.fc(hidden[-1,:])


# In[15]:


model = Model(1,64)


# In[16]:


optimizer=torch.optim.Adam(model.parameters(), lr=0.001)


# In[17]:


loss_fn = nn.MSELoss()


# In[18]:


num_epochs=100


# In[19]:


for epoch in range(num_epochs):
    output = model(train_x)
    loss=loss_fn(output,train_y)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10==0 and epoch !=0:
        print(epoch , "epoch loss", loss.detach().numpy())


# In[20]:


model.eval()
with torch.no_grad():
    output=model(test_x)


# In[21]:


pred=mm.inverse_transform(output.numpy())
real=mm.inverse_transform(test_y.numpy())


# In[22]:


plt.plot(pred.squeeze(),color="red",label="predicted")
plt.plot(real.squeeze(),color="green",label="real")
plt.show()


# In[ ]:




