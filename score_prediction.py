#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import pandas as pd
import pickle


# In[3]:


# Loading the dataset
df = pd.read_csv('C:\\Users\\LOHANI\\Desktop\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\nidhi\\ipl.csv')


# In[4]:


df.head()


# In[5]:


# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = [ 'mid','venue','batsman', 'bowler', 'strike_rate_batsmen','striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)


# In[6]:


df.head()


# In[7]:


df.dtypes


# In[8]:


# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]


# In[9]:


# Removing the first 5 overs data in every match
df = df[df['overs']>=5.0]


# In[10]:


# Converting the column 'date' from string into datetime object
from datetime import datetime
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%d-%m-%Y'))


# In[11]:


df.head()


# In[12]:


# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])


# In[13]:


encoded_df.columns


# In[14]:


encoded_df.to_csv('C:\\Users\\LOHANI\\Desktop\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\encoded_df.csv')


# In[15]:


# Rearranging the columns
encoded_df = encoded_df[['date','bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
              'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
              'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
              'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
              'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]


# In[16]:


# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]


# In[17]:


y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values


# In[18]:


# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)


# In[19]:


# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)


# In[27]:


# Creating a pickle file for the classifier
filename = 'C:\\Users\\LOHANI\\Desktop\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\nidhi\\first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))


# # Ridge regression

# In[20]:


## Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


# In[21]:


ridge=Ridge()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)
ridge_regressor.fit(X_train,y_train)


# In[22]:


print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)


# In[23]:


prediction=ridge_regressor.predict(X_test)


# In[24]:


import seaborn as sns
sns.distplot(y_test-prediction)


# In[25]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))


# # Lasso regression

# In[26]:


from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV


# In[27]:


lasso=Lasso()
parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40]}
lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)

lasso_regressor.fit(X_train,y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)


# In[28]:


prediction1=lasso_regressor.predict(X_test)


# In[29]:


import seaborn as sns
sns.distplot(y_test-prediction1)


# In[30]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, prediction1))
print('MSE:', metrics.mean_squared_error(y_test, prediction1))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction1)))


# In[31]:


# Creating a pickle file for the classifier
filename = 'C:\\Users\\LOHANI\\Desktop\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\IPL-First-Innings-Score-Prediction-Deployment-master\\nidhi\\first-innings-score-lasso-model.pkl'
pickle.dump(lasso_regressor, open(filename, 'wb'))


# In[ ]:




