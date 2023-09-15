#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries and Visualizing the data

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/91944/OneDrive/Documents/Bharat Intern/HousePricePrediction.csv")
 
print(df.head(5))


# In[3]:


obj = (df.dtypes == 'object')
object_cols = list(obj[obj].index)
print("Categorical variables:",len(object_cols))

int_ = (df.dtypes == 'int')
num_cols = list(int_[int_].index)
print("Integer variables:",len(num_cols))

fl = (df.dtypes == 'float')
fl_cols = list(fl[fl].index)
print("Float variables:",len(fl_cols))


# In[4]:


plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(),
cmap = 'BrBG',
fmt = '.2f',
linewidths = 2,
annot = True)


# In[5]:


df.shape


# In[6]:


plt.figure(figsize=(18, 36))
plt.title('Categorical Features: Distribution')
plt.xticks(rotation=90)
index = 1
 
for col in object_cols:
    y = df[col].value_counts()
    plt.subplot(11, 4, index)
    plt.xticks(rotation=90)
    sns.barplot(x=list(y.index), y=y)
    index += 1


# # Data Preprocessing

# In[10]:


df.drop(['Id'],
        axis=1,
        inplace=True)
#as "ID column is not used in predictions we make, we can remove"

df['SalePrice'] = df['SalePrice'].fillna(df['SalePrice'].mean())
#replacing all the empty valules of the salary coloumn with its mean value to maintain a distributed data


# In[64]:


from sklearn.preprocessing import OneHotEncoder

s = (new_df.dtypes == 'object')
object_cols = list(s[s].index) #we get all features which have the object's datatype
print("Categorical variables:")
print(object_cols)
print('No. of. categorical features: ',len(object_cols))
#One hot Encoding is to convert categorical data into binary vectors. 


# In[17]:


#we take list of all the features and apply OneHotEncoding.
OH_encoder = OneHotEncoder(sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(new_df[object_cols]))
OH_cols.index = new_df.index
OH_cols.columns = OH_encoder.get_feature_names()
df_final = new_df.drop(object_cols, axis=1)
df_final = pd.concat([df_final, OH_cols], axis=1)


# # Data Splitting and model traning

# In[18]:


from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

X = df_final.drop(['SalePrice'], axis=1)
Y = df_final['SalePrice']

# Split the training set into
# training and validation set
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, train_size=0.8, test_size=0.2, random_state=0)
#we take 80 percent of data for traning and remaining 20 percent for testing.


# In[55]:


#Form sklearn library, we import LinearRegression 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
 
model_LR = LinearRegression()
model_LR.fit(X_train, Y_train)
Y_pred = model_LR.predict(X_valid)
 
print(mean_absolute_percentage_error(Y_valid, Y_pred))


# In[63]:



plt.scatter(range(len(Y_valid)), Y_valid, label='Actual values', marker='+',)

# Plot Predicted Values as a Line
plt.plot(range(len(Y_valid)), Y_pred, label='Predicted values', color='yellow',linewidth=2.6)

plt.xlabel('Data Point Index')
plt.ylabel('Sale Price')
plt.legend()
plt.title('Actual vs. Predicted Sale Price')
plt.show()

