#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing libraries for data analysis
import numpy as np
import pandas as pd
import random as rnd

#importing libraries for data visualisation
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#importing libraries for machine learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[3]:


#Data Extraction from movie dataset
df_movie= pd.read_csv(r'C:/Users/HP/Desktop/movies.csv' ,sep ='::' ,engine ='python')
df_movie.columns =['MovieIDs','moviename','category']
df_movie.dropna(inplace=True)
df_movie.head()


# In[4]:


#extracting data from the rating dataset
df_rating =pd.read_csv(r'C:/Users/HP/Desktop/ratings.csv', sep ='::',engine ='python')
df_rating.columns=['ID','MovieID','rating','timestamp']
df_rating.dropna(inplace=True)
df_rating.head()


# In[6]:


#extracting data ffrom user dataset
df_user=pd.read_csv(r'C:/Users/HP/Desktop/users.csv', sep ='::',engine='python')
df_user.columns=['userid','gender','age','occupation','zip-code']
df_user.dropna()
df_user.head()


# In[7]:


df=pd.concat([df_movie,df_rating,df_user],axis=1)
df.head()


# performing exploratory data analysis

# In[8]:


#visualizing user age distribution
df['age'].value_counts().plot(kind='barh',alpha=0.7,figsize=(10,10))
plt.show()


# In[10]:


df.age.plot.hist(bins=25)
plt.title("Distribution of users age")
plt.ylabel('count of users')
plt.xlabel('Age')


# In[12]:


labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79']
df['age_group'] = pd.cut(df.age, range(0, 81, 10), right=False, labels=labels)
df[['age', 'age_group']].drop_duplicates()[:10]


# In[16]:


#visualizing the overall rating of user
df['rating'].value_counts().plot(kind='bar',alpha=0.7,figsize=(10,10))
plt.show()


# In[19]:


groupedby_movieName=df.groupby('moviename')
groupedby_rating=df.groupby('rating')
grouped_uid=df.groupby('userid')


# In[20]:


movies = df.groupby('moviename').size().sort_values(ascending=True)[:1000]
print(movies)


# In[21]:


ToyStory_data = groupedby_movieName.get_group('Toy Story 2 (1999)')
ToyStory_data.shape


# In[22]:


#Find and visualize the user rating of the movie “Toy Story”
plt.figure(figsize=(10,10))
plt.scatter(ToyStory_data['moviename'],ToyStory_data['rating'])
plt.title('Plot showing  the user rating of the movie “Toy Story”')
plt.show()


# In[23]:


#Find and visualize the viewership of the movie “Toy Story” by age group
ToyStory_data[['moviename','age_group']]


# In[24]:


#Find and visualize the top 25 movies by viewership rating
top_25 = df[25:]
top_25['rating'].value_counts().plot(kind='barh',alpha=0.6,figsize=(7,7))
plt.show()


# In[26]:


#Visualize the rating data by user of user id = 2696
userid_2696 = grouped_uid.get_group(2696)
userid_2696[['userid','rating']]


# performin machine learning on forst 500 extracted records

# In[27]:


#First 500 extracted records
first_500 = df[500:]
first_500.dropna(inplace=True)


# In[29]:


#Use the following features:movie id,age,occupation
features = first_500[['MovieIDs','age','occupation']].values


# In[31]:


#Use rating as label
labels = first_500[['rating']].values


# In[32]:


#Create train and test data set
train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)


# In[33]:


#Create a histogram for movie
df.age.plot.hist(bins=25)
plt.title("Movie & Rating")
plt.ylabel('MovieID')
plt.xlabel('Ratings')


# In[34]:


#Create a histogram for age
df.age.plot.hist(bins=25)
plt.title("Age & Rating")
plt.ylabel('Age')
plt.xlabel('Ratings')


# In[35]:


#Create a histogram for occupation
df.age.plot.hist(bins=25)
plt.title("Occupation & Rating")
plt.ylabel('Occupation')
plt.xlabel('Ratings')


# In[36]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train, train_labels)
Y_pred = logreg.predict(test)
acc_log = round(logreg.score(train, train_labels) * 100, 2)
acc_log


# In[37]:


# Support Vector Machines

svc = SVC()
svc.fit(train, train_labels)
Y_pred = svc.predict(test)
acc_svc = round(svc.score(train, train_labels) * 100, 2)
acc_svc


# In[38]:


# K Nearest Neighbors Classifier

knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train, train_labels)
Y_pred = knn.predict(test)
acc_knn = round(knn.score(train, train_labels) * 100, 2)
acc_knn


# In[39]:


# Gaussian Naive Bayes

gaussian = GaussianNB()
gaussian.fit(train, train_labels)
Y_pred = gaussian.predict(test)
acc_gaussian = round(gaussian.score(train, train_labels) * 100, 2)
acc_gaussian


# In[40]:


# Perceptron

perceptron = Perceptron()
perceptron.fit(train, train_labels)
Y_pred = perceptron.predict(test)
acc_perceptron = round(perceptron.score(train, train_labels) * 100, 2)
acc_perceptron


# In[41]:


# Linear SVC

linear_svc = LinearSVC()
linear_svc.fit(train, train_labels)
Y_pred = linear_svc.predict(test)
acc_linear_svc = round(linear_svc.score(train, train_labels) * 100, 2)
acc_linear_svc


# In[42]:


# Stochastic Gradient Descent

sgd = SGDClassifier()
sgd.fit(train, train_labels)
Y_pred = sgd.predict(test)
acc_sgd = round(sgd.score(train, train_labels) * 100, 2)
acc_sgd


# In[43]:


# Decision Tree

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train, train_labels)
Y_pred = decision_tree.predict(test)
acc_decision_tree = round(decision_tree.score(train, train_labels) * 100, 2)
acc_decision_tree


# In[44]:


# Random Forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train, train_labels)
Y_pred = random_forest.predict(test)
random_forest.score(train, train_labels)
acc_random_forest = round(random_forest.score(train, train_labels) * 100, 2)
acc_random_forest


# In[45]:


models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


# In[ ]:




