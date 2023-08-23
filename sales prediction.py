#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Data Visualisation
import matplotlib.pyplot as plt 
import seaborn as sns


# In[2]:


advertising = pd.DataFrame(pd.read_csv(r'C:/Users/HP/Desktop/advertising.csv'))
advertising.head()


# In[3]:


#data inspection
advertising.shape


# In[4]:


advertising.info()


# In[5]:


advertising.describe()


# #### data cleaning

# In[6]:


# Checking Null values
advertising.isnull().sum()*100/advertising.shape[0]
# There are no NULL values in the dataset, hence it is clean.


# In[7]:


# Outlier Analysis
fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(advertising['TV'], ax = axs[0])
plt2 = sns.boxplot(advertising['Newspaper'], ax = axs[1])
plt3 = sns.boxplot(advertising['Radio'], ax = axs[2])
plt.tight_layout()


# #### exploratory data analysis

# In[8]:


sns.boxplot(advertising['Sales'])
plt.show()


# In[9]:


# Let's see how Sales are related with other variables using scatter plot.
sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()


# In[10]:


# Let's see the correlation between different variables.
sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()


# #### model building

# In[11]:


X = advertising['TV']
y = advertising['Sales']


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[13]:


X_train.head()


# In[14]:


y_train.head()


# #### building a linear model

# In[15]:


import statsmodels.api as sm


# In[16]:


# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()


# In[17]:


lr.params


# In[18]:


print(lr.summary())


# In[19]:


plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()


# #### model evaluation 

# In[20]:


y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)


# In[21]:


fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()


# In[22]:


plt.scatter(X_train,res)
plt.show()


# In[23]:


# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)


# In[24]:


y_pred.head()


# In[25]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[26]:


np.sqrt(mean_squared_error(y_test, y_pred))


# In[27]:


r_squared = r2_score(y_test, y_pred)
r_squared


# In[28]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()


# In[ ]:




