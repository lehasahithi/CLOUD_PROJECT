#!/usr/bin/env python
# coding: utf-8

# In[66]:


import pandas as pd
import boto3
import io

# Replace with your actual S3 bucket name and file key
s3_bucket = 'climate143'   # Your S3 bucket name
file_key = 'cloud_dataset.csv'  # The path to the file in S3

# Set up the S3 client
s3_client = boto3.client('s3')

try:
    # Load the data directly from S3
    obj = s3_client.get_object(Bucket=s3_bucket, Key=file_key)
    raw_data = obj['Body'].read()
    
    # Try reading with different encodings
    try:
        climate_data = pd.read_csv(io.BytesIO(raw_data), encoding='utf-8')
    except UnicodeDecodeError:
        print("Failed to decode with UTF-8, trying ISO-8859-1...")
        climate_data = pd.read_csv(io.BytesIO(raw_data), encoding='ISO-8859-1')
    
    # Display the first few rows to verify data loading
    print(climate_data.head())

except Exception as e:
    print(f"Error loading the dataset: {e}")


# In[67]:


# Check the structure of the data
print(climate_data.info())

# View summary statistics for numerical columns
print(climate_data.describe())


# In[68]:


# Display count of missing values in each column
print("Missing values per column:")
print(climate_data.isnull().sum())


# In[69]:


climate_data.head()


# In[70]:


# Plot temperature over time
plt.figure(figsize=(12, 6))
plt.plot(climate_data['Temperature at 2 Meters (C) '], color='blue', label='Temperature at 2 Meters (T2M)')
plt.title("Temperature Over Time")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.show()


# In[71]:


climate_data.head()


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming climate_data is already defined and contains your dataset

# Calculate the correlation matrix
correlation_matrix = climate_data.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Climate Variables")
plt.show()


# In[73]:


# Calculate temperature difference from dew point

# Rename specific columns
climate_data.rename(columns={'Temperature at 2 Meters (C) ' : 'Temperature_at_2_Meters', 
                             'Dew/Frost Point at 2 Meters ©': 'Dew/Frost_Point_at_2_Meters'}, inplace=True)

# Display the updated column names
print(climate_data.columns)


# In[85]:


from sklearn.preprocessing import StandardScaler

# Define numerical columns to scale
numerical_cols = [
    'Wind Speed at 2 Meters (m/s) ', 'Temperature_at_2_Meters', 
    'Dew/Frost_Point_at_2_Meters', 'Earth Skin Temperature (C) ', 
    'Temperature at 2 Meters Range ©', 'Surface Pressure (kPa) ', 
    'All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day)', 
    'Clear Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day) ', 
    'CLRSKY_SFC_SW_DWN', 'All Sky Surface PAR Total (W/m^2) ', 
    'Precipitation Corrected (mm/day) ', 'Relative Humidity at 2 Meters (%)'
]
scaler = StandardScaler()
climate_data[numerical_cols] = scaler.fit_transform(climate_data[numerical_cols])


# In[86]:


from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = climate_data.drop(['Temperature_at_2_Meters'], axis=1)
y = climate_data['Temperature_at_2_Meters']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[88]:


import boto3

# Save processed data locally
X_train.to_csv('X_train.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Upload to S3
bucket_name = 'climate143'
s3 = boto3.client('s3')
s3.upload_file('X_train.csv', bucket_name, 'preprocessed/X_train.csv')
s3.upload_file('y_train.csv', bucket_name, 'preprocessed/y_train.csv')
s3.upload_file('X_test.csv', bucket_name, 'preprocessed/X_test.csv')
s3.upload_file('y_test.csv', bucket_name, 'preprocessed/y_test.csv')


# In[89]:


import sagemaker
from sagemaker import get_execution_role
from sagemaker.inputs import TrainingInput

# Define SageMaker session and role
role = get_execution_role()
sagemaker_session = sagemaker.Session()

# Define data paths
train_input = TrainingInput(s3_data=f's3://{bucket_name}/preprocessed/X_train.csv', content_type="csv")
output_path = f's3://{bucket_name}/output/'

# Set up the XGBoost estimator
xgboost_container = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1")
xgb = sagemaker.estimator.Estimator(
    image_uri=xgboost_container,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    output_path=output_path,
    sagemaker_session=sagemaker_session
)

# Set model hyperparameters
xgb.set_hyperparameters(objective="reg:squarederror", num_round=100)


# In[90]:


# Start model training
xgb.fit({"train": train_input})


# In[91]:


# Deploy model to endpoint
predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m5.large')


# In[101]:


# Download test data from S3
s3.download_file(bucket_name, 'preprocessed/X_test.csv', 'X_test.csv')
s3.download_file(bucket_name, 'preprocessed/y_test.csv', 'y_test.csv')
X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')

# Ensure that X_test has the correct number of features
expected_feature_count = 14  # Replace with the actual expected feature count

# Drop any extra columns, if present
if X_test.shape[1] > expected_feature_count:
    X_test = X_test.iloc[:, :expected_feature_count]

# Convert X_test to CSV format for prediction
csv_data = X_test.to_csv(index=False, header=False).encode('utf-8')

# Make predictions on test data
predictions = predictor.predict(csv_data, initial_args={"ContentType": "text/csv"}).decode('utf-8').splitlines()

# Split each prediction line by commas and convert to floats
predictions = [float(pred) for line in predictions for pred in line.split(",")]



# In[ ]:


from sklearn.metrics import mean_squared_error, mean_absolute_error

# Calculate metrics
rmse = root_mean_squared_error(y_test, predictions, squared=False)
mae = mean_absolute_error(y_test, predictions)
print(f"RMSE: {rmse}, MAE: {mae}")


# In[ ]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Define the model
model = RandomForestRegressor()

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

# Get the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Score (MSE):", -grid_search.best_score_)


# In[ ]:


# Retrieve the best model from GridSearchCV or RandomizedSearchCV
best_model = grid_search.best_estimator_  # or random_search.best_estimator_

# Train on the training data
best_model.fit(X_train, y_train)

# Make predictions on the test set
final_predictions = best_model.predict(X_test)

# Evaluate with metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test, final_predictions)
mae = mean_absolute_error(y_test, final_predictions)
r2 = r2_score(y_test, final_predictions)

print(f"Final Model Evaluation:\nMSE: {mse}\nMAE: {mae}\nR²: {r2}")


# In[ ]:




