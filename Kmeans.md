# K-Means Clustering on Sales Data

## 1. Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
```

## 2. Load the Dataset

```python
# latin1 encoding is needed for this file
df = pd.read_csv("sales_data_sample.csv", encoding="latin1")
print("Shape:", df.shape)
```

## 3. Select Features and Scale Them

```python
features = ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]
X = df[features].dropna()
X_scaled = StandardScaler().fit_transform(X)
```

## 4. Elbow Method

```python
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("Elbow Method")
plt.grid()
plt.show()
```

## 5. Apply K-Means

```python
# Set k from the elbow in the plot above
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
X["Cluster"] = kmeans.fit_predict(X_scaled)
```

## 6. Analyze the Clusters

```python
print("\nCluster sizes:\n", X["Cluster"].value_counts())
print("\nCluster averages:\n", X.groupby("Cluster").mean())
```

## 7. Visualize the Clusters

```python
plt.scatter(X["QUANTITYORDERED"], X["SALES"], c=X["Cluster"], cmap="viridis")
plt.xlabel("Quantity Ordered")
plt.ylabel("Sales")
plt.title("K-Means Clusters")
plt.show()
```