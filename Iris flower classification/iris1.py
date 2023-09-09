#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set(style="white", color_codes=True)


# In[42]:


iris = pd.read_csv("./Desktop/Iris1.csv")
iris.head()


# In[3]:


iris["Species"].value_counts()


# In[4]:


sns.FacetGrid(iris, hue="Species",height=6).map(plt.scatter, "PetalLengthCm", "PetalWidthCm").add_legend()


# Logistic Regression

# In[6]:


flower_mapping = {'setosa': 0,'versicolor': 1,'virginica':2}
iris["Species"] = iris["Species"].map(flower_mapping)


# In[7]:


iris.head()


# In[9]:


X=iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].values
y=iris[['Species']].values 


# ###  Logistic Regression

# In[12]:


from sklearn.linear_model import LogisticRegression


# In[13]:


model = LogisticRegression()


# In[14]:


model.fit(X, y)


# ### Accuracy

# In[15]:


model.score(X,y)


# In[16]:


expected = y
predicted = model.predict(X)
predicted


# ### summarize the fit of the model

# In[18]:


from sklearn import metrics


# In[19]:


print(metrics.classification_report(expected, predicted))


# In[20]:


print(metrics.confusion_matrix(expected, predicted))


# ### Regularization

# In[21]:


model = LogisticRegression(C=20,penalty='l2' )


# In[22]:


model.fit(X,y)


# In[23]:


model.score(X,y)


# In[24]:


iris.isnull().sum()


# ## Visualizing the dataset

# In[25]:


plt.boxplot(iris['SepalLengthCm'])


# In[26]:


plt.boxplot(iris['SepalWidthCm'])


# In[27]:


plt.boxplot(iris['PetalLengthCm'])


# In[28]:


plt.boxplot(iris['PetalWidthCm'])


# In[64]:


"""import matplotlib.colors as mcolors
#sns.heatmap(iris.corr())
custom_colors = ['#00EEEE','#8DEEEE','#79CDCD','#528B8B']  # Red, Green, Blue

# Create a custom colormap using custom colors
custom_cmap = mcolors.LinearSegmentedColormap.from_list("Custom Colormap", custom_colors)

# Create a heatmap with the custom colormap
sns.heatmap(iris.corr(), cmap=custom_cmap)  """

# Calculate the correlation matrix using NumPy
corr_matrix = np.corrcoef(X, rowvar=False)

# Define custom colors for the heatmap
custom_colors = ['#8DEEEE','#00CDCD','#79CDCD','#528B8B']  # Teal shades

# Create a custom colormap using custom colors
custom_cmap = mcolors.LinearSegmentedColormap.from_list("Custom Colormap", custom_colors)

# Create a heatmap with the custom colormap
sns.heatmap(corr_matrix, cmap=custom_cmap)

# Show the heatmap
import matplotlib.pyplot as plt
plt.show()


# # Summarize the fit of the mode

# In[40]:


print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# # using scikit-learn library

# In[48]:


# Import necessary libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# In[50]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[51]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data (optional but recommended)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)


# In[ ]:




