# Uber Ride Price Prediction

## 1. Importing Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

## 2. Data Loading and Pre-processing
```python
# Load dataset and drop missing values
df = pd.read_csv("uber.csv").dropna()

# Extract datetime features
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], errors='coerce')
for feature in ['hour', 'day', 'month', 'year', 'dayofweek']:
    df[feature] = getattr(df['pickup_datetime'].dt, feature)
```

## 3. Outlier Detection
```python
# Filter unrealistic fares and geographical coordinates
df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100) &
        (df['pickup_latitude'].between(-90, 90)) & (df['dropoff_latitude'].between(-90, 90)) &
        (df['pickup_longitude'].between(-180, 180)) & (df['dropoff_longitude'].between(-180, 180))]

# Vectorized Haversine distance calculation
lon1, lat1, lon2, lat2 = map(np.radians, [df['pickup_longitude'], df['pickup_latitude'], 
                                         df['dropoff_longitude'], df['dropoff_latitude']])
dlon, dlat = lon2 - lon1, lat2 - lat1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
df['distance_km'] = 2 * 6371 * np.arcsin(np.sqrt(a))

# Remove zero-distance trips
df = df[df['distance_km'] > 0]
```

## 4. Correlation Analysis
```python
# Generate heatmap
plt.figure(figsize=(8, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()
```

## 5. Model Implementation (Linear Regression & Random Forest)
```python
# Define features and target variable
features = ['distance_km', 'hour', 'day', 'month', 'year', 'dayofweek']
X_train, X_test, y_train, y_test = train_test_split(df[features], df['fare_amount'], test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
}
```

## 6. Model Evaluation and Comparison
```python
# Train, predict, and evaluate
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    
    results.append({
        'Model': name,
        'R2': r2_score(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'MAE': mean_absolute_error(y_test, preds)
    })

# Display evaluation metrics
res_df = pd.DataFrame(results)
print("\nModel Comparison Table:")
print(res_df.to_string(index=False))

# Plot performance metrics comparison
res_df.plot(x='Model', y='R2', kind='bar', legend=False, rot=0, figsize=(6,4))
plt.ylabel("R² Score")
plt.title("Model Comparison")
plt.show()
```
