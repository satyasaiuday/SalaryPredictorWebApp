#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


# In[2]:


dataset=pd.read_csv(r"C:\Users\Prabhas\Downloads\Simple-Regression-using-LR-and-DT-main\Simple-Regression-using-LR-and-DT-main\Salary_Data.csv")


# In[3]:


dataset.head(7)


# In[4]:


plt.scatter(dataset['YearsExperience'],dataset['Salary'])
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()


# In[5]:


plt.bar(dataset['YearsExperience'],dataset['Salary'])
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()


# In[6]:


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
X,Y


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=101)


# In[9]:


X_train


# In[10]:


X_test


# In[11]:


Y_train


# In[12]:


Y_test


# # Linear Regression

# In[13]:


from sklearn.linear_model import LinearRegression


# In[14]:


LR=LinearRegression()


# In[15]:


LR.fit(X_train,Y_train)


# In[16]:


X_train


# In[17]:


Y_train


# In[18]:


pickle.dump(LR,open('model.pkl','wb'))


# In[19]:


model = pickle.load(open('model.pkl','rb'))


# In[21]:


print(model.predict([[2]]))

