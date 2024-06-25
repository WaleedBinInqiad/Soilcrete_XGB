#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


df=pd.read_csv("CS.csv")
df.head()


# In[3]:


x = df.iloc[: ,:-1] #get a copy of dataset exclude last column
y = df.iloc[: ,-1:]#get array of dataset in column 1st


# In[4]:


x.head()


# In[5]:


y.head()


# In[6]:


df_description = df.describe()
df_description


# In[7]:


corr=df.corr()
corr.head()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=1)


# In[10]:


regressor= MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=0.1,
    gamma=1,
    max_depth=20
))


# In[11]:


model=regressor.fit(x_train, y_train)


# In[12]:


model= model.fit(x, y)


# In[13]:


y_train_pred=model.predict(x_train)


# In[14]:


y_test_pred=model.predict(x_test)


# In[15]:


import sklearn.metrics as sm
print("RFR Mean absolute error =", round(sm.mean_absolute_error(y_train, y_train_pred),5)) 
print("MLR Mean squared error =", round(sm.mean_squared_error(y_train, y_train_pred),5)) 
print("MLR Median absolute error =", round(sm.median_absolute_error(y_train, y_train_pred), 5)) 
print("MLR Explain variance score =", round(sm.explained_variance_score(y_train, y_train_pred), 5)) 
print("MLR R2 score =", round(sm.r2_score(y_train,y_train_pred ), 5))


# In[16]:


import sklearn.metrics as sm
print("RFR Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred),5)) 
print("MLR Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred),5)) 
print("MLR Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 5)) 
print("MLR Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 5)) 
print("MLR R2 score =", round(sm.r2_score(y_test, y_test_pred ), 5))


# In[18]:


new_input = np.array([[0.4, 0, 50, 2, 4070]])
new_preds = model.predict(new_input)
print("New input predictions: ", new_preds)


# In[ ]:





# In[ ]:





# In[ ]:






# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




