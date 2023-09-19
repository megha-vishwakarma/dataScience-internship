#!/usr/bin/env python
# coding: utf-8

# # Sales Prediction using Python

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("./Desktop/SIP/advertising.csv")
df.head()


# In[4]:


df.shape


# In[5]:


df.describe()

Basic Observation

Avg expense spend is highest on TV

Avg expense spend is lowest on Radio

Max sale is 27 and min is 1.6
# In[6]:


sns.pairplot(df, x_vars=['TV', 'Radio','Newspaper'], y_vars='Sales', kind='scatter')
plt.show()


# **Pair Plot Observation :**
# *When advertising cost increases in TV Ads the sales will increase as well. While the for newspaper and radio it is bit unpredictable.

# In[31]:


df['TV'].plot.hist(bins=10)


# In[64]:


df['Radio'].plot.hist(color=["#4F94CD"], bins=10, xlabel="Radio");


# In[67]:


df['Newspaper'].plot.hist(color=["#00C5CD"],bins=10, xlabel="newspaper");


# **Histogram Observation**
# 
# * The majority sales is the result of low advertising cost in newspaper

# In[108]:


df = pd.read_csv("./Desktop/SIP/sales.csv")


# In[38]:


# Specify the colormap ('Blues' for shades of blue)

sns.heatmap(df.corr(), annot=True, cmap='Blues')
# plt.figure(figsize=(12, 10))
#sns.heatmap(df.corr(),annot = True)
plt.show()


# In[29]:


fig,ax = plt.subplots(2,2 ,figsize = (25,15))
for i,subplot in zip(df,ax.flatten()):
    sns.boxplot(df[i],ax = subplot)


# * SALES IS HIGHLY COORELATED WITH THE TV
# 
# ### Lets train our model using linear regression as it is coorelated with only one variable TV

# In[80]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['TV']], df[['Sales']], test_size = 0.3,random_state=0)


# In[81]:


print(X_train)


# In[82]:


print(y_train)


# In[84]:


print(X_test)


# In[85]:


print(y_test)


# In[86]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)


# In[87]:


res= model.predict(X_test)
print(res)


# In[88]:


model.coef_


# In[89]:


model.intercept_


# In[90]:


0.05473199* 69.2 + 7.14382225


# In[94]:


plt.plot(res);


# In[36]:


plt.scatter(X_test, y_test)
plt.plot(X_test, 7.14382225 + 0.05473199 * X_test, 'g')
plt.show()


# In[100]:


y_train_pred = model.predict(X_train)
res = (y_train - y_train)
res


# In[14]:


data = pd.read_csv("./Desktop/SIP/sales.csv")
df.head()
x=data.drop(['Sales'],axis=1)


# In[15]:


y=data['Sales']


# In[16]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)


# In[17]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[18]:


# Linear Regression Model
import statsmodels.formula.api as sm

lr_model = sm.ols(formula="Sales ~ TV + Radio + Newspaper", data=data).fit()


# In[19]:


# Print the coefficients of the linear regression model
print(lr_model.params, "\n")

THE LIEAR REGRESSION EQUATION IS : Y=2.93888+(0.045765TV)+(0.188530Radio)-(0.001037*Newspaper)
# In[20]:


# Print the summary of the linear regression model
print(lr_model.summary())


# In[21]:


#MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train,y_train)


# In[22]:


X_train_scaled = scaler.transform(x_train)
X_test_scaled = scaler.transform(x_test)


# In[23]:


from sklearn.linear_model import LinearRegression, Ridge,Lasso

models = [
    ("Linear Regression", LinearRegression()),
    ("Ridge Regression", Ridge()),
]


# In[24]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

for name, model in models:
    model.fit(X_train_scaled,y_train)
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    print("")
    print("{}".format(name))
    print("Mean Squared Error: {}".format(mse))
    print("R2 Score: {}".format(r2))
    print("Cross-Validation R2: {}".format(cv_scores.mean()))


# In[25]:


df1=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})


# In[26]:


sns.lmplot(x="Actual",y="Predicted",data=df1,fit_reg=False)
d_line=np.arange(df1.min().min(),df1.max().max())
plt.plot(d_line,d_line,color="red",linestyle="-")
plt.show()


# ## Prediction using test data

# In[27]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[32]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y, test_size=0.2,random_state=42)


# In[34]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Create a dictionary of regression models
regression_models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "Decision Tree Regressor": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}

# Lists to store evaluation metrics
mse_scores = []
mae_scores = []
r2_scores = []

# Train and evaluate regression models
for model_name, model in regression_models.items():
    model.fit(xtrain, ytrain)
    y_pred = model.predict(xtest)

    # Calculate evaluation metrics
    mse = mean_squared_error(ytest, y_pred)
    mae = mean_absolute_error(ytest, y_pred)
    r2 = r2_score(ytest, y_pred)

    # Append scores to lists
    mse_scores.append(mse)
    mae_scores.append(mae)
    r2_scores.append(r2)


# In[35]:


# Create a DataFrame to store the results
regression_scores_df = pd.DataFrame({
    "Algorithm": regression_models.keys(),
    "Mean Squared Error (MSE)": mse_scores,
    "Mean Absolute Error (MAE)": mae_scores,
    "R-squared (R2)": r2_scores
})

# Print the regression scores DataFrame
regression_scores_df


# In[28]:


TV= float(input("Enter the TV value: "))
Radio = float(input("Enter the Radio value: "))
Newspaper= float(input("Enter the Newspaper value: "))

new_data = pd.DataFrame({
    "TV": [TV],
    "Radio": [Radio],
    "Newspaper": [Newspaper]
})

print("-------------------------------------")
new_pred = lr.predict(new_data)
print("Predicted Sales : {}".format(abs(new_pred)))


# #### Concluding with saying that above mention solution is successfully able to predict the sales using advertising platform datasets

# In[ ]:




