#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import h5py
import dask.dataframe as dd
from HDF5Dataset import HDF5Dataset
import joblib
import numpy as np
import json
from dask_ml.preprocessing import RobustScaler
import matplotlib.pyplot as plt
from architecture.RNN import RNN
import gc


# In sequential architecture, current hidden state is a function of the current input and previous hidden state:
# 
# 
# ### h(t) = f(h(t-1), x(t); W)
# 
# W are the parameters of function (in our case NN)
#  

# For RNN:
# 
# a(t) = W * h(t-1) + U * x(t) + b1
# h(t) = tanh(a(t))
# o(t) = V * h(t) + b2

# In[2]:


# Initialise device

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Primary device set to GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Primary device set to CPU.")


# In[3]:


def pandas_sequence_generator(input_df, window_size, stride=1, batch_size=20000):
    df_array = input_df.values
    start = 0
    total_length = len(df_array)
    sequences = []
    outputs = []
    while start < total_length - window_size:
        sequences.append(df_array[start : start + window_size])
        outputs.append(df_array[start+window_size])
        start += stride
        
        if len(sequences) >= batch_size:
            yield np.array(sequences), np.array(outputs)
            sequences = []
            outputs = []
    
    if len(sequences) > 0:
        yield np.array(sequences), np.array(outputs)
        
def dask_sequence_generator(input_df, window_size, stride=1, batch_size=20000):
    # Convert Dask DataFrame to Dask Array for easier slicing
    df_array = input_df.to_dask_array(lengths=True)

    # Compute total length using the .shape attribute of the Dask Array
    total_length = df_array.shape[0]
    
    # Initialize start index and lists for sequences and outputs
    start = 0
    sequences = []
    outputs = []
    
    while start < total_length - window_size:
        end = start + window_size
        # Append the slice of the array (all columns in the window)
        sequences.append(df_array[start:end].compute())  # Compute necessary for yielding numpy arrays
        
        # Outputs could be the next row or specific columns depending on the task
        outputs.append(df_array[end].compute())  # Compute the next point
        
        start += stride
        
        # Yield batch when enough sequences have been collected
        if len(sequences) >= batch_size:
            yield np.array(sequences), np.array(outputs)
            sequences = []
            outputs = []
    
    # Yield any remaining sequences after the loop
    if len(sequences) > 0:
        yield np.array(sequences), np.array(outputs)



# In[4]:


def write_to_hdf5(input_df, window_size, stride_size, batch_size, storage_path, dataset_name, 
                  label_name = 'label'):
    sequence_data_size = int(np.floor((len(input_df) - window_size) / stride_size ))
    num_features = input_df.shape[1]  # Number of features (columns) in the DataFrame
    
    gen = dask_sequence_generator(input_df, window_size, stride_size, batch_size)
    
    with h5py.File(storage_path, 'w') as f:
        # Create a dataset with pre-allocated memory for sequences and features
        dset = f.create_dataset(dataset_name, (sequence_data_size, window_size, num_features), dtype='float32')
        y_set = f.create_dataset(label_name, sequence_data_size)
        count = 0
        
        for batch in gen:
            features = batch[0]
            y = batch[1]
            num_data = features.shape[0]
            dset[count:count + num_data] = features
            y_set[count: count + num_data] = np.squeeze(y)
            count += num_data


# In[5]:


# Since there are some corrupt values, we read it as object/string and then convert it to float. 
# Reading invalid values as float is throwing error in Dask so using this approach to get around the issue
df = dd.read_csv('data/daily-minimum-temperatures-in-me.csv', dtype={'Daily minimum temperatures': 'object'})

# Convert the column to numeric float16
df['Daily minimum temperatures'] = dd.to_numeric(df['Daily minimum temperatures'], errors='coerce')
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].astype('float16')

# Use map_partitions to apply the pandas interpolate method to each partition
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].map_partitions(
    lambda s: s.interpolate(method='linear'), meta=('x', 'float16'))


# In[6]:


# Apply scaler and also save the columns and their order
df = df.set_index('Date', drop=True)
display(df.head())
column_order = list(df.columns)
print(column_order)
with open('meta/model_column_order.json', 'w') as f:
    json.dump(column_order, f)
    
# Create and fit the scaler
scaler = RobustScaler()
scaler.fit(df)

scaled_df = scaler.transform(df)


# In[7]:


# Find diff between max and min
print(df['Daily minimum temperatures'].min().compute())
print((scaled_df['Daily minimum temperatures'].max() - df['Daily minimum temperatures'].min()).compute())


# In[8]:


window_size = 30
stride_size = 1
batch_size = 32
write_to_hdf5(scaled_df[['Daily minimum temperatures']], window_size,
              stride_size, batch_size, 'meta/sequence.h5', 'sequences')

partitioned_df = scaled_df.repartition(npartitions = 5)
write_to_hdf5(partitioned_df[['Daily minimum temperatures']], window_size,
              stride_size, batch_size, 'meta/partitioned_sequence.h5', 'sequences')



# In[9]:


display(df.head(5))


# In[10]:


display(partitioned_df.head(5))


# In[11]:


## Check if partitioned dataframe has any affect on file generated

file_path = 'meta/sequence.h5'
hdf5_file = h5py.File(file_path, 'r')
data = hdf5_file['sequences']

file_path = 'meta/partitioned_sequence.h5'
p_hdf5_file = h5py.File(file_path, 'r')
p_data = p_hdf5_file['sequences']

print(data.shape)
print(p_data.shape)

assert len(data) == len(p_data)

hdf5_file.close()
p_hdf5_file.close()


# In[12]:


# Open the original HDF5 file
file_path = 'meta/partitioned_sequence.h5'
hdf5_file = h5py.File(file_path, 'r')  # Open in read-only mode
data = hdf5_file['sequences']
labels = hdf5_file['label']

# Create new HDF5 files for training and testing data
train_file = h5py.File('meta/train_data.h5', 'w')
test_file = h5py.File('meta/test_data.h5', 'w')

# Create datasets for data in the new files
train_dataset = train_file.create_dataset('data', (0,) + data.shape[1:], maxshape=(None,) + data.shape[1:], dtype=data.dtype)
test_dataset = test_file.create_dataset('data', (0,) + data.shape[1:], maxshape=(None,) + data.shape[1:], dtype=data.dtype)

# Create datasets for labels in the new files
train_labels = train_file.create_dataset('label', (0,) + labels.shape[1:], maxshape=(None,) + labels.shape[1:], dtype=labels.dtype)
test_labels = test_file.create_dataset('label', (0,) + labels.shape[1:], maxshape=(None,) + labels.shape[1:], dtype=labels.dtype)


# In[13]:


train_segment_size = 90
test_segment_size = 10
total_samples = data.shape[0]

i = 0
while i < total_samples:
    end_train = min(i + train_segment_size, total_samples)
    if end_train > i:  # Check if there is data to process
        train_dataset.resize(train_dataset.shape[0] + (end_train - i), axis=0)
        train_dataset[-(end_train - i):] = data[i:end_train]
        train_labels.resize(train_labels.shape[0] + (end_train - i), axis=0)
        train_labels[-(end_train - i):] = labels[i:end_train]

    i = end_train
    end_test = min(i + test_segment_size, total_samples)
    if end_test > i:  # Check if there is data to process
        test_dataset.resize(test_dataset.shape[0] + (end_test - i), axis=0)
        test_dataset[-(end_test - i):] = data[i:end_test]
        test_labels.resize(test_labels.shape[0] + (end_test - i), axis=0)
        test_labels[-(end_test - i):] = labels[i:end_test]

    i = end_test

train_file.close()
test_file.close()
hdf5_file.close()


# In[23]:


# Assume the paths to your HDF5 files
train_file_path = 'meta/train_data.h5'
test_file_path = 'meta/test_data.h5'

# Create dataset instances
train_dataset = HDF5Dataset(train_file_path, 'data', 'label')
test_dataset = HDF5Dataset(test_file_path, 'data', 'label')




# In[24]:


# Save scaler to disk
joblib.dump(scaler, 'meta/model_robust_scaler.joblib')


# In[25]:


model = RNN(1, 10, 1, 2)
mse_loss = nn.MSELoss()
adam_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, 'min')


# In[26]:


# Create DataLoader instances
train_loader_arr = []
train_loader = DataLoader(train_dataset, batch_size = 64, num_workers = 4, shuffle = True)
ns_train_loader = DataLoader(train_dataset, batch_size = 64, num_workers = 4, shuffle = False)

train_loader_arr.append(train_loader)
train_loader_arr.append(ns_train_loader)


test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

num_epochs = 26

for epoch in range(num_epochs):
    loss_arr = []
    for data, labels in train_loader_arr[1]:
        adam_optimizer.zero_grad()
        res = model.forward(data)
        out = res[:, -1, :]
        loss = mse_loss(out, labels.view(-1, 1))
        
        loss_arr.append(loss.item())
        loss.backward()
        adam_optimizer.step()
        
    test_loss_arr = []
    with torch.no_grad():
        for data, labels in test_loader:
            res = model.forward(data)
            out = res[:, -1, :]
            loss = mse_loss(out, labels.view(-1, 1))
            test_loss_arr.append(loss.item())
    
    print("Epoch - ", epoch, " loss: ",np.mean(loss_arr), " test loss: ",np.mean(test_loss_arr))
    


# In[51]:


# train_dataset.__getitem__(0)[0]


# In[27]:


del train_loader
del ns_train_loader
del train_dataset
del test_dataset
gc.collect()


# In[19]:


get_ipython().system('jupyter nbconvert --to script RNN-from-scratch.ipynb')

h5py.__version__


# In[20]:


loss_sum = 0
model_basic = RNN(1, 3, 1, 2)
actual = []
pred = []

with torch.no_grad():
    for data, labels in test_loader:
        res = model.forward(data)
        out = res[:, -1, :]
        # print(scaler.inverse_transform(out), scaler.inverse_transform(labels))
        actual.extend(scaler.inverse_transform(labels).detach().numpy())
        pred.extend(scaler.inverse_transform(out).detach().numpy())
        loss = mse_loss(out, labels.view(-1, 1))
        loss_sum += loss.item()
    
print(loss_sum)


# In[21]:


plt.plot(actual)
plt.plot(pred)
plt.show()


# In[22]:


err_arr = []
for pair in zip(actual, pred):
    err_arr.append(abs((pair[0] - pair[1])[0]))
print(100*np.mean(err_arr)/np.mean(actual), " % error on average")


# In[ ]:




