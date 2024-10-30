import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('diabetes.csv')
# df.info()

# #---check for null values---
# print("Nulls")
# print("=====")
# print(df.isnull().sum())

# #---check for 0s---
# print("0s")
# print("==")
# print(df.eq(0).sum())

df[['Glucose','BloodPressure','SkinThickness',
 'Insulin','BMI','DiabetesPedigreeFunction','Age']] = \
 df[['Glucose','BloodPressure','SkinThickness',
 'Insulin','BMI','DiabetesPedigreeFunction','Age']].replace(0,np.NaN)

df.fillna(df.mean(), inplace = True) # replace NaN with the mean
# print(df.eq(0).sum())

corr = df.corr()
# print(corr)

# fig, ax = plt.subplots(figsize=(10, 10))
# cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
# fig.colorbar(cax)
# ticks = np.arange(0,len(df.columns),1)
# ax.set_xticks(ticks)
# ax.set_xticklabels(df.columns)
# plt.xticks(rotation = 90)
# ax.set_yticklabels(df.columns)
# ax.set_yticks(ticks)
# #---print the correlation factor---
# for i in range(df.shape[1]):
#  for j in range(9):
#     text = ax.text(j, i, round(corr.iloc[i][j],2),ha="center", va="center", color="w")
# plt.show()

# import seaborn as sns
# sns.heatmap(df.corr(),annot=True)
# #---get a reference to the current figure and set its size---
# fig = plt.gcf()
# fig.set_size_inches(8,8)
# print(df.corr().nlargest(4, 'Outcome').index)

from sklearn import linear_model
from sklearn.model_selection import cross_val_score
#---features---
X = df[['Glucose','BMI','Age']]
#---label---
y = df.iloc[:,8]
log_regress = linear_model.LogisticRegression()
log_regress_score = cross_val_score(log_regress, X, y, cv=10, 
scoring='accuracy').mean()
print("Logistic: ", log_regress_score)

result = []
result.append(log_regress_score)

from sklearn.neighbors import KNeighborsClassifier
#---empty list that will hold cv (cross-validates) scores---
cv_scores = []
#---number of folds---
folds = 10
#---creating odd list of K for KNN---
ks = list(range(1,int(len(X) * ((folds - 1)/folds)), 2))
#---perform k-fold cross validation---
for k in ks:
 knn = KNeighborsClassifier(n_neighbors=k)
 score = cross_val_score(knn, X, y, cv=folds, scoring='accuracy').mean()
 cv_scores.append(score)
#---get the maximum score---
knn_score = max(cv_scores)
#---find the optimal k that gives the highest score---
optimal_k = ks[cv_scores.index(knn_score)]
print(f"The optimal number of neighbors is {optimal_k}")
print(knn_score)
result.append(knn_score)

from sklearn import svm
linear_svm = svm.SVC(kernel='linear')
linear_svm_score = cross_val_score(linear_svm, X, y,
 cv=10, scoring='accuracy').mean()
print(linear_svm_score)
result.append(linear_svm_score)

rbf = svm.SVC(kernel='rbf')
rbf_score = cross_val_score(rbf, X, y, cv=10, scoring='accuracy').mean()
print(rbf_score)
result.append(rbf_score)

algorithms = ["Logistic Regression", "K Nearest Neighbors", "SVM Linear Kernel", "SVM RBF Kernel"]
cv_mean = pd.DataFrame(result,index = algorithms)
cv_mean.columns=["Accuracy"]
cv_mean_sorted =  cv_mean.sort_values(by="Accuracy",ascending=False)

print(cv_mean_sorted)