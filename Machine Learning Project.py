#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


import numpy as np


# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


model=LinearRegression()


# In[6]:


data=pd.read_csv("C:/Users/DELL/Desktop/project/train-data.csv")


# In[7]:


model.fit(data[["Year","Kilometers_Driven","Fuel_Type","Transmission","Owner_Type","Mileage","Engine in CC","Power in bhp","Seats"]],data["Price in Lakh"])


# In[8]:


X=data.drop(['Car Company','Price in Lakh','Location'],axis=1,inplace=False)


# In[9]:


X.head()


# In[10]:


y=data.drop([  "Car Company" ,"Location" ,"Year", "Kilometers_Driven" , "Fuel_Type" , "Transmission" , "Owner_Type" , "Mileage" , "Engine in CC" , "Power in bhp" , "Seats" ], axis = 1 , inplace = False)


# In[11]:


from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split( X , y , test_size = 0.2 , random_state = 0 )


# In[12]:


X_train


# In[13]:


y_train


# In[14]:


model.fit( X_train , y_train)


# In[15]:


X_test


# In[16]:


y_test


# In[17]:


y_pred = model.predict(X_test)
y_pred


# In[18]:


X_test=[[2010,26246,1,0,1,15.29,1591,121.3,5]]
y_pred=model.predict(X_test)
print(y_pred)


# In[ ]:





# In[ ]:




