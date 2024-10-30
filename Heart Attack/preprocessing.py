import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('heart.csv')
df.info()

# df.fillna(df.mean(), inplace = True)

#---check for null values---
print("Nulls")
print("=====")
print(df.isnull().sum())


#---check for 0s---
print("0s")
print("==")
print(df.eq(0).sum())

corr = df.corr()
print(corr)


fig, ax = plt.subplots(figsize=(10, 10))
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(df.columns),1)
ax.set_xticks(ticks)
ax.set_xticklabels(df.columns)
plt.xticks(rotation = 90)
ax.set_yticklabels(df.columns)
ax.set_yticks(ticks)
#---print the correlation factor---
for i in range(df.shape[1]):
 for j in range(14):
    text = ax.text(j, i, round(corr.iloc[i][j],2),
    ha="center", va="center", color="w")
# plt.show()

X = df[['cp', 'thalachh', 'slp']]
y = df.iloc[:,13]
# print("X: ", X)
print("y: ", y)

result = []
#---Logistic Regression---
log_regress = linear_model.LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, y, cv=10, scoring='accuracy').mean()
result.append(log_regress_score)
print(log_regress_score)


#---KNN---
cv_scores = []
folds = 10
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
for k in ks:
 knn = KNeighborsClassifier(n_neighbors=k)
 score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
 cv_scores.append(score)
knn_score = max(cv_scores)
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)
result.append(knn_score)

#---Linear SVM---
from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y,
 cv=10, scoring='accuracy').mean()
print(linear_svm_score)
result.append(linear_svm_score)

#---RBF SVM---
rbf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
print(rbf_score)

result.append(rbf_score)
algorithms = ["Logistic Regression", "K Nearest Neighbors", "SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result,index = algorithms)
cv_mean.columns=["Accuracy"]
cv_mean.sort_values(by="Accuracy",ascending=False)
print(cv_mean)