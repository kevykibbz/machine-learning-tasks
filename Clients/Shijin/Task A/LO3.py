#!/usr/bin/env python
# coding: utf-8

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pandas as pd


# In[8]:


# Load the cereals dataset
cereals_data = pd.read_csv('cereals.csv')

# Drop rows with NaN values
cereals_data.dropna(inplace=True)


# In[9]:


# Select features and target variable
X = cereals_data[['Calories', 'Protein', 'Fat', 'Sodium', 'Fiber', 'Carbo', 'Sugars', 'Potass', 'Vitamins']]
y = cereals_data['Rating']  # Assuming 'Rating' is the target variable

# Discretize the target variable 'Rating' into two classes: High and Low
# Define a threshold to categorize ratings
threshold = cereals_data['Rating'].mean()

# Map ratings above the threshold as 'High' and ratings below or equal to the threshold as 'Low'
cereals_data['Rating_Class'] = ['High' if rating > threshold else 'Low' for rating in cereals_data['Rating']]

# Update the target variable
y = cereals_data['Rating_Class']


# In[10]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)


# In[6]:


# Choose a classification model (e.g., Logistic Regression)
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train_imputed, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_imputed)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[ ]:




