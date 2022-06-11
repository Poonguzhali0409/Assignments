#!/usr/bin/env python
# coding: utf-8

# In[348]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.xgboost import XGBRegressor


# In[245]:


train = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\train.csv')
test = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\test.csv')
sample_sub = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\sample_submission.csv')


# In[246]:


train[:5]


# In[247]:


train.shape


# In[248]:


test.shape


# In[249]:


test[:5]


# In[250]:


sample_sub[:5]


# In[251]:


train.info()


# In[252]:


train.describe(include='all').T


# In[253]:


train['Age'].unique()


# In[254]:


train['source'] = 'train'
test['source'] = 'test'


# In[255]:


train.head()


# In[256]:



data = pd.concat([train,test])


# In[257]:


data.head()


# In[258]:


data.isna().sum()


# In[259]:


train.info()


# In[260]:


df = data.copy()
df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])


# In[261]:


df.head()


# In[262]:


data.head()


# In[263]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()


# In[264]:


df['Gender'] = lr.fit_transform(df['Gender'])
df['Age'] = lr.fit_transform(df['Age'])
df['City_Category'] = lr.fit_transform(df['City_Category'])


# In[265]:


df.head()


# In[266]:



df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')


# In[267]:


df.isnull().sum()


# In[268]:


df.info()


# In[269]:


with_header = df.copy()   
df = df.drop(["User_ID","Product_ID"],axis=1)


# In[299]:


test_preprocessed.head()


# In[271]:



train_preprocessed = df[df['source']=='train'].drop(columns='source')
test_preprocessed = df[df['source']=='test'].drop(columns=['source', 'Purchase'])


# In[272]:


train_preprocessed.to_csv("train_preprocessed.csv", index = False)
test_preprocessed.to_csv("test_preprocessed.csv", index = False)


# ### MODELLING

# In[273]:


train_pre = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\train_preprocessed.csv')
test_pre = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\test_preprocessed.csv')
sample_sub = pd.read_csv (r'C:\Users\COMPUER\Desktop\DataScience\sample_submission.csv')


# In[274]:


test_pre


# In[275]:


train_pre.head()


# In[276]:


x =  train_pre.drop(columns = ["Purchase"])
x


# In[277]:


y = train_pre['Purchase']


# In[278]:


train_X, test_X, train_y, test_y = train_test_split(x,y, test_size = 0.2, random_state=32)


# In[279]:


model = LinearRegression()
model.fit(train_X, train_y)


# In[280]:


model.coef_


# In[281]:


model.intercept_


# In[282]:


train_X_pred = model.predict(train_X)
test_X_pred = model.predict(test_X)


# In[300]:


train_X_RMSE = np.sqrt(mean_squared_error(train_y, train_X_pred))
test_X_RMSE = np.sqrt(mean_squared_error(test_y, test_X_pred))
print("Train RMSE : ",train_X_RMSE)
print("Test RMSE : ",test_X_RMSE)
train_X_R2 = r2_score(train_y, train_X_pred)
test_X_R2 = r2_score(test_y, test_X_pred)
print("Train R2 : ",train_X_R2)
print("Test R2 : ",test_X_R2)


# In[284]:


mean_absolute_error(test_y, y_pred)


# In[285]:


mean_squared_error(test_y, y_pred)


# In[286]:


r2_score(test_y, y_pred)


# In[287]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(test_y, y_pred)))


# In[288]:


train_X_pred = model.predict(train_X)
test_X_pred = model.predict(test_X)


# In[333]:


test_p = with_header[with_header['source']=='test'].drop(columns=['User_ID', 'Product_ID','source','Purchase'])
test_f =  with_header[with_header['source']=='test'].drop(columns=['source','Purchase'])


# In[327]:


with_header.head()


# In[328]:


test_p.head()test_p.head()


# In[329]:


test_pred = model.predict(test_p)


# In[330]:


test_pred


# In[334]:


test_f['Purchase'] = test_pred


# In[335]:


mody_output = test_f[['User_ID', 'Product_ID','Purchase']]


# In[339]:


mody_output.to_csv("result_modify.csv", index = False)


# In[345]:





# In[346]:





# In[ ]:




