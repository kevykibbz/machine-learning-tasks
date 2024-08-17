#!/usr/bin/env python
# coding: utf-8

# In[10]:


# 1. Linear Regression

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import pandas as pd


# In[11]:


# Load the cereals dataset
cereals_data = pd.read_csv('cereals.csv')

# Drop rows with NaN values
cereals_data.dropna(inplace=True)

# Taking 'X' as our feature matrix and 'y' as our target variable
X = cereals_data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber', 'Carbo', 'Sugars', 'Potass', 'Vitamins']]
y = cereals_data['Rating']  # Taking 'Rating' as the target variable to predict



# In[12]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a Linear Regression model
model = LinearRegression()


# In[13]:


# Fit the model to the training data
model.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Evaluate the model using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


# In[18]:


# 2. Classification

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pandas as pd

# Load your dataset
cereals_data = pd.read_csv('cereals.csv')

# Drop rows with NaN values
cereals_data.dropna(inplace=True)

# Taking 'X' as our feature matrix and 'y' as our target variable
X = cereals_data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber', 'Carbo', 'Sugars', 'Potass', 'Vitamins']]
y = cereals_data['Rating']  # Taking 'Rating' as the target variable to predict

# Defining bins for low, medium, and high ratings
bins = [0, 50, 75, 100]  # Define your own bins based on your criteria
labels = ['low', 'medium', 'high']

# Map ratings to classes
cereals_data['Rating_Class'] = pd.cut(cereals_data['Rating'], bins=bins, labels=labels)

# Modify target variable 'y' to use the new Rating_Class
y = cereals_data['Rating_Class']


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose 'median' or 'most_frequent' as well
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Fit the model to the training data
model.fit(X_train_imputed, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test_imputed)

# Evaluate the model using accuracy score and classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[20]:


# 3. Clustering 

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# Load your dataset
cereals_data = pd.read_csv('cereals.csv')

# Drop rows with NaN values
cereals_data.dropna(inplace=True)

# Select relevant features for clustering
# Choose features that you believe are important for clustering cereals
# For example, you can select 'Calories', 'Protein', 'Fat', 'Sodium', etc.
X = cereals_data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber', 'Carbo', 'Sugars', 'Potass', 'Vitamins']]

# Choose the number of clusters (k) based on domain knowledge or techniques like the elbow method
k = 3

# Create a KMeans model
model = KMeans(n_clusters=k, random_state=42)

# Fit the model to the data
model.fit(X)

# Predict the cluster labels
labels = model.labels_

# Evaluate the clustering using silhouette score
silhouette = silhouette_score(X, labels)
print("Silhouette Score:", silhouette)


# In[ ]:




