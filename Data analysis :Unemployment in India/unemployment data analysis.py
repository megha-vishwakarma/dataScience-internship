#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[27]:


df=pd.read_csv("./Desktop/unEmployment/Unemployment in India.csv")
df.head()


# In[3]:


df.info()    #overall data


# In[4]:


df.shape

df.dropna()  #Removal of missing values


# In[5]:


#check for duplicates
df.duplicated().sum()


# In[28]:


#Removing duplicate values
df.drop_duplicates()


# In[7]:


df.columns


# ## Data visualisation

# In[19]:


plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
displot = sns.displot(df[' Estimated Unemployment Rate (%)'])
displot.set(title='Unemployment Rate Distribution')
plt.show()


# In[10]:


#sns.barplot(df['Region'],df[' Estimated Unemployment Rate (%)'])

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have loaded your data into a DataFrame named 'df'

plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
sns.barplot(data=df, x='Region', y=' Estimated Unemployment Rate (%)')
plt.title('Unemployment Rate by Region')
plt.xlabel('Region')
plt.ylabel('Estimated Unemployment Rate (%)')
plt.xticks(rotation=45)  # Rotate x-axis labels if needed for readability
plt.show()


# In[84]:


#Unemployment in different states 
plt.title('Indian Unemployment')
sns.histplot(x=' Estimated Unemployment Rate (%)',hue='Region',data=df)
#plt.legend(loc='upper right')
plt.show()


# In[49]:


import matplotlib.pyplot as plt

# Assuming you have loaded your data into a DataFrame named 'df'
area_counts = df['Area'].value_counts()

# Get the labels and counts for the pie chart
labels = area_counts.index
counts = area_counts.values
colors = ['#5CACEE', '#00E5EE']

# Plot the pie chart
plt.figure(figsize=(7, 5))
plt.pie(counts, labels=labels, colors=colors,autopct='%1.1f%%',shadow = True)
plt.title('Distribution of Areas')
plt.show()


# In[80]:


# Unemployment in different states
plt.figure(figsize=(12, 6))  # Adjust the figure size as needed
plt.title('Indian estimated Employment (statewise)')

# Use sns.histplot with horizontal orientation and specify y for the State column
sns.histplot(y='State', x='Employed', data=df, hue='State', multiple='stack', shrink=0.8, element='bars')

plt.xlabel('Employed')
plt.ylabel('State')
# plt.legend().remove()  # Remove the legend
plt.show()


# In[94]:


df=pd.read_csv("./Desktop/unEmployment/un.csv")
df.head()
print(df.info())


# In[97]:


# Visualize the time series data
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Unemployment Rate'])
plt.title('Estimated Unemployment Rate Over Time')
plt.xlabel('Date')
plt.ylabel('Unemployment Rate')
plt.grid(True)
plt.show()

# Time series decomposition (optional)
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(df['Unemployment Rate'], model='multiplicative')
result.plot()
plt.show()


# In[ ]:




