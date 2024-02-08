#!/usr/bin/env python
# coding: utf-8

# #Multiple Linear Regression

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error


# In[2]:


#Load the dataset
df=pd.read_csv("advertising.csv")


# In[3]:


df.head()


# In[4]:


#feature and target selection
X=df[["TV","Radio"]]
y=df[["Sales"]]


# In[5]:


#splitting dataset
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[6]:


#model
lr=LinearRegression()


# In[7]:


#training the model
lr.fit(X_train,y_train)


# In[8]:


lr.intercept_


# In[9]:


lr.coef_


# In[10]:


#testing with training dataset only
y_pred=lr.predict(X_test)


# In[12]:


#Evaluation
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(rmse)
print(mae)
print(r2)

