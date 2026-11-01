# K-Nearest Neighbors on Diabetes Dataset

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
```

## 2. Load the Dataset

```python
df = pd.read_csv("diabetes.csv")
print("Shape:", df.shape)
print("Diabetic (1) vs Non-diabetic (0):\n", df["Outcome"].value_counts())
```

## 3. Pre-process the Data

```python
# Zeros in these columns are invalid, so replace them with the median
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
df[cols] = df[cols].fillna(df[cols].median())
```

## 4. Separate Features and Label

```python
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
```

## 5. Split into Training and Testing Sets

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## 6. Scale the Features

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## 7. Implement K-Nearest Neighbors

```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

## 8. Compute the Metrics

```python
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nConfusion Matrix:\n", cm)
print("Accuracy  :", round(accuracy, 4))
print("Error Rate:", round(error_rate, 4))
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
```