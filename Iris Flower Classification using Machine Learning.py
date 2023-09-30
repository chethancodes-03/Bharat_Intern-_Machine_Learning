#!/usr/bin/env python
# coding: utf-8

# ###  Iris Dataset classification based on Petal and Sepal Length using Machine Learning

# In[3]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the Iris dataset
df = sns.load_dataset("iris")

# Rename the columns
df.columns = columns

# Display the first few rows of the dataset
df.head()


# In[14]:


# Some basic statistical analysis about the data
df.describe()


# In[15]:


# Visualize the whole dataset
sns.pairplot(df, hue='Class_labels')


# In[16]:


# Separate features and target which are sepal and petal lengths.
data = df.values
X = data[:,0:4] 
Y = data[:,4]


# In[17]:


# Calculate average of each features for all classes
Y_Data = np.array([np.average(X[:, i][Y==j].astype('float32')) for i in range (X.shape[1]) for j in (np.unique(Y))])
Y_Data_reshaped = Y_Data.reshape(4, 3)
Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
X_axis = np.arange(len(columns)-1)
width = 0.25


# In[18]:


# Plot the average
plt.bar(X_axis, Y_Data_reshaped[0], width, label = 'Setosa')
plt.bar(X_axis+width, Y_Data_reshaped[1], width, label = 'Versicolour')
plt.bar(X_axis+width*2, Y_Data_reshaped[2], width, label = 'Virginica')
plt.xticks(X_axis, columns[:4])
plt.xlabel("Features")
plt.ylabel("Value in cm.")
plt.legend(bbox_to_anchor=(1.3,1))
plt.show()


# In[20]:


# Split the data to train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[21]:


# Support vector machine algorithm used for classification as its precise and balanced.
from sklearn.svm import SVC
svn = SVC()
svn.fit(X_train, y_train)


# In[22]:


# Predict from the test dataset
predictions = svn.predict(X_test)

# Calculate the accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)


# In[23]:


#evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))


# In[24]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]]) # testing of model with custom input.

#Prediction of the species from the input vector
prediction = svn.predict(X_new)
print("Prediction of Species: {}".format(prediction))

