#!/usr/bin/env python
# coding: utf-8

# #### IMPORTING LIBRARIES
# 

# In[1]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
import random
from scipy import stats
import warnings
from warnings import filterwarnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.preprocessing import LabelEncoder,PolynomialFeatures,StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict,GridSearchCV,KFold,train_test_split
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import joblib
from sklearn.svm import SVC,SVR


# In[6]:


df=pd.read_csv(r'C:/Users/HP/Desktop/creditcard.csv')


# #### EXPLORATORY DATA ANALYSIS

# In[7]:


df.shape


# In[8]:


df.head()


# In[9]:


df.describe()


# In[10]:


df.info()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.isna().sum()


# In[13]:


df["Hour"] = df["Time"].apply(lambda x: np.ceil(float(x)/3600) % 24)
df["Hour"] = df["Hour"].astype("int")


# #### CHECKING THE DISTRIBUTION OF THE TARGET VARIABLE

# In[15]:


print(df['Class'].value_counts()) # Count the occurrences of fraud (1) and non-fraud (0)
sns.countplot(df['Class']) # Plot a bar plot to visualize the distribution
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()


# #### VISUALIZING THE DISTRIBUTION OF NUMERIC FEATURES

# In[16]:


plt.figure(figsize=(10, 6))
sns.distplot(df['Hour'])
plt.xlabel('Hour')
plt.ylabel('Density')
plt.title('Distribution of Time')
plt.show()


# In[17]:


plt.figure(figsize=(10, 6))
sns.distplot(df['Amount'])
plt.xlabel('Amount')
plt.ylabel('Density')
plt.title('Distribution of Amount')
plt.show()


# #### COMPARING THE DISTRIBUTION OF NUMERIC FEATURES FOR FRAUD AND NON FRAUD CASES

# In[18]:


fraud_cases = df[df['Class'] == 1]
non_fraud_cases = df[df['Class'] == 0]

plt.figure(figsize=(10, 6))
sns.distplot(fraud_cases['Amount'], label='Fraud')
sns.distplot(non_fraud_cases['Amount'], label='Non-Fraud')
plt.xlabel('Amount')
plt.legend()
plt.title('Distribution of Amount for Fraud and Non-Fraud Cases')
plt.show()


# #### VISUALIZING THE COREELATION FEATURES USINH HEATMAP

# In[19]:


plt.figure(figsize=(20, 15))
sns.heatmap(df.corr(), cmap='summer_r', annot=True, fmt='.2f', linewidths=0.2)
plt.title('Correlation Heatmap')
plt.show()


# In[21]:


plt.figure(figsize=(20,15))
sns.boxplot(data=df[["Class", "Amount", "Time"]], orient="h")
plt.show()


# In[22]:


sns.pairplot(df[['Hour','Class', 'Amount']], diag_kind="hist")


# #### PREPROCESS THE DATA

# In[23]:


scaler = StandardScaler()
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

df.drop(['Time', 'Amount'], axis=1, inplace=True)


# In[25]:


# Split the dataset into train and test sets
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# #### TRAINING THE MACHINE LEARNING MODEL

# In[27]:


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# #### EVALUATING THE MODEL 

# In[28]:


y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))


# In[30]:


print(classification_report(y_test,y_pred))


# In[ ]:




