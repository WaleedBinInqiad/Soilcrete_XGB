#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


# In[20]:


df=pd.read_csv("E.csv")
df.head()


# In[21]:


x = df.iloc[: ,:-1] #get a copy of dataset exclude last column
y = df.iloc[: ,-1:]#get array of dataset in column 1st


# In[22]:


x.head()


# In[23]:


y.head()


# In[24]:


df_description = df.describe()
df_description


# In[25]:


corr=df.corr()
corr.head()


# In[26]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=1)


# In[27]:


regressor= MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=100,
    reg_lambda=0.1,
    gamma=1,
    max_depth=10
))


# In[28]:


model=regressor.fit(x_train, y_train)


# In[29]:


model= model.fit(x, y)


# In[30]:


y_train_pred=model.predict(x_train)


# In[31]:


y_test_pred=model.predict(x_test)


# In[32]:


import sklearn.metrics as sm
print("RFR Mean absolute error =", round(sm.mean_absolute_error(y_train, y_train_pred),5)) 
print("MLR Mean squared error =", round(sm.mean_squared_error(y_train, y_train_pred),5)) 
print("MLR Median absolute error =", round(sm.median_absolute_error(y_train, y_train_pred), 5)) 
print("MLR Explain variance score =", round(sm.explained_variance_score(y_train, y_train_pred), 5)) 
print("MLR R2 score =", round(sm.r2_score(y_train,y_train_pred ), 5))


# In[33]:


import sklearn.metrics as sm
print("RFR Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred),5)) 
print("MLR Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred),5)) 
print("MLR Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 5)) 
print("MLR Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 5)) 
print("MLR R2 score =", round(sm.r2_score(y_test, y_test_pred ), 5))


# In[34]:


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




