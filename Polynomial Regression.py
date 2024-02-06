#!/usr/bin/env python
# coding: utf-8

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


df=pd.read_csv("advertising.csv")


# In[3]:


df.head()


# In[4]:


from sklearn.preprocessing import PolynomialFeatures


# In[6]:


pf=PolynomialFeatures(2)
#degree of X is 2


# In[9]:


#selecting feature and target
X=df[["TV","Radio"]]
y=df[["Sales"]]


# In[8]:


#spilt test and train
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[11]:


#convert training and test features to quadratic one
x_train_poly=pf.fit_transform(X_train)#training dataset
x_test_poly=pf.transform(X_test)#testing model


# In[12]:


#model building and training
lr=LinearRegression()
lr.fit(x_train_poly,y_train)


# In[13]:


#testing
y_pred=lr.predict(x_test_poly)


# In[14]:


#evaluation
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[15]:


r2=r2_score(y_test,y_pred)


# In[16]:


r2


# In[ ]:




