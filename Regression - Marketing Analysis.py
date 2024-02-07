#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
df = pd.read_csv('advertising.csv')
df.head()


# In[4]:


X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[9]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()


# In[10]:


lr.fit(X_train, y_train)


# In[11]:


y_pred = lr.predict(X_test)


# In[14]:


from sklearn import metrics
import numpy as np
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




