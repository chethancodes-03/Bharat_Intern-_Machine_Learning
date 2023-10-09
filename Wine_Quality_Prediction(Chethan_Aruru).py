#!/usr/bin/env python
# coding: utf-8

# # Wine prediction using linear regression Machine Learning

# In[1]:


#import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("C:/Users/91944/OneDrive/Documents/Bharat Intern/archive (4)/WineQT.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe().T


# In[5]:


df.isnull().sum() #no null values.


# In[6]:


df.hist(bins=20, figsize=(10, 10))
plt.show()


# In[7]:


#plot to visualise the number data for each quality of wine.
plt.bar(df['quality'], df['alcohol'])
plt.xlabel('quality')
plt.ylabel('alcohol')
plt.show()


# In[8]:


df['best quality'] = [1 if x > 5 else 0 for x in df.quality]
df.replace({'white': 1, 'red': 0}, inplace=True)


# In[9]:


from sklearn.model_selection import train_test_split
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=40)

X_train.shape, X_test.shape


# In[10]:


from sklearn.preprocessing import MinMaxScaler
norm = MinMaxScaler()
X_train = norm.fit_transform(X_train)
X_test = norm.transform(X_test)


# In[17]:


from sklearn import metrics
from sklearn.linear_model import LinearRegression
# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)


# In[18]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred) b

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared (R2) Score: {r2}')

