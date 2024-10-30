import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('data_combined.csv')

# Fill missing values with the mean of each column
df.fillna(df.mean(), inplace=True)

# Features: ChuyenCan, BTL, GiuaKi
X = df[['ChuyenCan', 'BTL', 'GiuaKi']]

# Target: CuoiKi (final exam score)
y = df['CuoiKi']

# Ensure the target is binary (0 or 1) for classification
# For demonstration, we'll binarize it based on a threshold (e.g., 7)
y_binary = np.where(y >= 7, 1, 0)  # Adjust threshold as needed

# Initialize the result list
result = []

# --- Logistic Regression ---
from sklearn.linear_model import LogisticRegression
log_regress = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
log_regress_score = cross_val_score(log_regress, X, y_binary, cv=10, scoring='accuracy').mean()
result.append(log_regress_score)
print(f"Logistic Regression Accuracy: {log_regress_score:.2f}")

# --- KNN ---
cv_scores = []
folds = 10
ks = list(range(1, int(len(X) * ((folds - 1) / folds)), 2))
for k in ks:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y_binary, cv=folds, scoring='accuracy').mean()
    cv_scores.append(score)
knn_score = max(cv_scores)
optimal_k = ks[cv_scores.index(knn_score)]
result.append(knn_score)
print(f"The optimal number of neighbors is {optimal_k}")
print(f"KNN Accuracy: {knn_score:.2f}")

# --- Linear SVM ---
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y_binary, cv=10, scoring='accuracy').mean()
result.append(linear_svm_score)
print(f"Linear SVM Accuracy: {linear_svm_score:.2f}")

# --- RBF SVM ---
rbf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(rbf, X, y_binary, cv=10, scoring='accuracy').mean()
result.append(rbf_score)
print(f"RBF SVM Accuracy: {rbf_score:.2f}")

# Summarize results
algorithms = ["Logistic Regression", "K Nearest Neighbors", "SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result, index=algorithms, columns=["Accuracy"])
cv_mean.sort_values(by="Accuracy", ascending=False, inplace=True)
print("\nModel Performance Summary:")
print(cv_mean)
