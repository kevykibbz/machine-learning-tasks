#!/usr/bin/env python
# coding: utf-8

# In[31]:


from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[4]:


# Dealing with different types of data - ordinal, categorical, encoding

# Sample of ordinal data
age_categories = ['child', 'teenager', 'young adult', 'adult', 'senior']

# Sample of categorical data
gender_categories = ['male', 'female', 'other']

# Example of encoding
label_encoder = LabelEncoder()
gender_encoded = label_encoder.fit_transform(['male', 'female', 'male', 'other'])


# In[21]:


# Collecting, storing, and making data ready for processing

# Collecting data using pandas
data = pd.read_csv('cereals.csv')

# Display the first few rows of the DataFrame
print(data.head())

# print(data.columns)
# print(data.dtypes)


# In[15]:


# Create the DataFrame
processed_data = pd.DataFrame({
    'Name': ['John Doe', 'Peter Johnson', 'Sarah Smith'],
    'Email Address': ['johndoe@gmail.com', 'peterjohnson@yahoo.com', 'sarah@gmail.com']
})

# Store data
processed_data.to_csv('processed_data',index=False)

# Read processed data
read_data=pd.read_csv('processed_data')

print(read_data.head())


# In[32]:


# Select only numeric columns
numeric_data = data.select_dtypes(include=['int64', 'float64'])

# Display the first few rows of the DataFrame
print(numeric_data.head())

# Example of making data ready for processing - feature scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)
scaled_data = scaled_data[~np.isnan(scaled_data).any(axis=1)]

print('-------------------------------------------------------')
print("Mean:", scaled_data.mean(axis=0))
print("Standard Deviation:", scaled_data.std(axis=0))
print("Minimum:", scaled_data.min(axis=0))
print("Maximum:", scaled_data.max(axis=0))


# In[34]:


# Histograms of scaled data
plt.figure(figsize=(10, 6))
for i in range(min(scaled_data.shape[1], 6)):  # Limit to 6 subplots
    plt.subplot(2, 3, i+1)
    plt.hist(scaled_data[:, i], bins=20)
    plt.title(f'Feature {i+1}')
plt.tight_layout()
plt.show()

# Box plots of scaled data
plt.figure(figsize=(10, 6))
plt.boxplot(scaled_data, labels=numeric_data.columns)
plt.title('Box Plot of Scaled Data')
plt.xticks(rotation=45)
plt.show()


# In[ ]:




