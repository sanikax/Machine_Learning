```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset (download emails.csv from Kaggle first)
df = pd.read_csv("emails.csv")
print("Shape:", df.shape)
print("Spam (1) vs Not Spam (0):\n", df["Prediction"].value_counts())

# 2. Separate features and label
X = df.drop(["Email No.", "Prediction"], axis=1)   # word-count features
y = df["Prediction"]                               # 0 = Not Spam, 1 = Spam

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# 6. Support Vector Machine
svm = SVC(kernel="linear")
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

# 7. Performance analysis
print("\n===== KNN =====")
print("Accuracy:", accuracy_score(y_test, knn_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred, target_names=["Not Spam", "Spam"]))

print("\n===== SVM =====")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred, target_names=["Not Spam", "Spam"]))

# 8. Comparison
print("\n===== Comparison =====")
print("KNN Accuracy:", round(accuracy_score(y_test, knn_pred) * 100, 2), "%")
print("SVM Accuracy:", round(accuracy_score(y_test, svm_pred) * 100, 2), "%")
```