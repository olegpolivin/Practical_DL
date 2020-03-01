#!/usr/bin/env python
# coding: utf-8

# # Homework: Not So Basic Artificial Neural Networks

# Your task is to implement a simple framework for convolutional neural networks training. While convolutional neural networks is a subject of lecture 3, we expect that there are a lot of students who are familiar with the topic.
# 
# In order to successfully pass this homework, you will have to:
# 
# - Implement all the blocks in `homework_modules.ipynb` (esp `Conv2d` and `MaxPool2d` layers). Good implementation should pass all the tests in `homework_test_modules.ipynb`.
# - Settle with a bit of math in `homework_differentiation.ipynb`
# - Train a CNN that has at least one `Conv2d` layer, `MaxPool2d` layer and `BatchNormalization` layer and achieves at least 97% accuracy on MNIST test set.
# 
# Feel free to use `homework_main-basic.ipynb` for debugging or as source of code snippets. 

# Note, that this homework requires sending **multiple** files, please do not forget to include all the files when sending to TA. The list of files:
# - This notebook with cnn trained
# - `homework_modules.ipynb`
# - `homework_differentiation.ipynb`

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt
from IPython import display


# In[2]:


# (re-)load layers
get_ipython().run_line_magic('run', 'homework_modules.ipynb')


# In[3]:


# batch generator
def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
        
    # Shuffle at the start of epoch
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        
        batch_idx = indices[start:end]
    
        yield X[batch_idx], Y[batch_idx]


# In[4]:


import mnist
X_train, y_train, X_val, y_val, X_test, y_test = mnist.load_dataset()  # your dataset


# In[5]:


# Your turn - train and evaluate conv neural network


# ### 1 Set up the neural network and train it

# In[6]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1)).toarray()
y_val_ohe = ohe.transform(y_val.reshape(-1, 1)).toarray()
y_test_ohe = ohe.transform(y_test.reshape(-1, 1)).toarray()


# In[7]:


# Optimizer
config = {
    'learning_rate' : 1e-3,
    'beta1' : 0.99,
    'beta2' : 0.999,
    'epsilon' : 1e-8
}


# In[8]:


# batch_size = 32
# batches = get_batches((X_train, y_train), batch_size)
# epochs = 5

net = Sequential()

# conv2d
conv_01 = Conv2d(1, 7, 3)
conv_02 = Conv2d(7, 14, 3)

# maxpool2d
maxpool_01 = MaxPool2d(2)
maxpool_02 = MaxPool2d(2)

# Linear layer
n_in = X_train.shape[1] * X_train.shape[2]
n_in = 686
linear_01 = Linear(n_in, 32)
linear_02 = Linear(32, 10)

# LogSoftmax layer
lsm = LogSoftMax()

# Stable negative likelihood criterion
criterion = ClassNLLCriterion()

net.add(conv_01) # bs x 7 x 28 x 28
net.add(ReLU())  # bs x 7 x 28 x 28
net.add(maxpool_01) # bs x 7 x 14 x 14
net.add(conv_02) # bs x 14 x 14 x 14
net.add(ReLU())    # bs x 14 x 14 x 14
net.add(maxpool_02) # bs x 14 x 7 x 7
net.add(Flatten()) # bs x 686
net.add(linear_01)
net.add(BatchNormalization(0.9))
net.add(ChannelwiseScaling(32))
net.add(ReLU())
net.add(Dropout())
net.add(linear_02)
net.add(BatchNormalization(0.9))
net.add(ChannelwiseScaling(10))
net.add(ReLU())
net.add(lsm)


# In[ ]:


print(net)


# Print here your accuracy on test set. It should be >97%. Don't forget to switch the network in 'evaluate' mode

# In[ ]:


# batches = get_batches((X_train, y_train),128)
loss_history = []
batch_size = 128
epochs = 10
# net.train()
for epoch in range(epochs):
    batches = get_batches((X_train, y_train_ohe), batch_size)
    k = 0
    for x_batch, y_batch in batches:
        net.zeroGradParameters()
        
        # Forward
        predictions = net.forward(x_batch)
        loss = criterion.forward(predictions, y_batch)
    
        # Backward
        dp = criterion.backward(predictions, y_batch)
        net.backward(x_batch, dp)
        
        # Update weights
        adam_optimizer(net.getParameters(),
                       net.getGradParameters(),
                       config,
                       {})       
        
        loss_history.append(loss)

        k = k + 1
        if k % 50 ==0:
            print(k)
        
    # Visualize
    display.clear_output(wait=True)
    plt.figure(figsize=(8, 6))
        
    plt.title("Training loss")
    plt.xlabel("#iteration")
    plt.ylabel("loss")
    plt.plot(loss_history, 'b')
    plt.show()
    
    net.evaluate()
    # Performance on validation set
    predictions_val = net.forward(X_val)
    loss_vall = criterion.forward(predictions_val, y_batch)
    pred_val = predictions_val.argmax(axis=1)
    acc_val = np.mean(pred_val == y_val)
    net.train()
    print(f'Validation loss/accuracy after epoch {epoch}: {loss_val}/{acc_val}\n')


# ### 2 Accuracy on test

# In[ ]:


net.evaluate()
pred_test = net.forward(X_test).argmax(axis=1)
acc_test = np.mean(pred_test == y_test)

print(f'Accuracy reached on test data: {acc_test}')

