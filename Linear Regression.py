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


df.shape


# In[5]:


df.info()
#datatypes all colms are float,all are non-null
#so there are no null values


# In[6]:


#cheking skweness
df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


#visu
plt.figure()
sns.scatterplot(data=df,x="TV",y="Sales")
plt.show()


# In[9]:


plt.figure()
sns.scatterplot(data=df,x="Radio",y="Sales")
plt.show()


# In[10]:


#checking linearity b/w
plt.figure()
sns.scatterplot(data=df,x="Newspaper",y="Sales")
plt.show()


# In[11]:


#heatmap to check correlation
plt.figure()
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[12]:


#selection of feature and target
X=df["TV"] #feature
y=df["Sales"]#target


# In[13]:


X=df["Newspaper"] #feature
y=df["Sales"]#targe


# In[14]:


X=df["Radio"] #feature
y=df["Sales"]#targe


# In[15]:


#split dataset into train and test
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)


# In[16]:


#maodel:Linear Regrassion model
lr=LinearRegression()


# In[18]:


lr.intercept_


# In[19]:


lr.coef_


# In[20]:


y_pred=lr.predict(np.array(X_test).reshape(-1,1))


# In[21]:


#Evalution of model
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y_test,y_pred)
print("MSE:- ",mse)
print("RMSE:- ",rmse)
print("MAE:- ",mae)


# In[22]:


r2=r2_score(y_test,y_pred)
r2

