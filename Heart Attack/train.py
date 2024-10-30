import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pickle

df = pd.read_csv('heart.csv')

X = df[['cp', 'thalachh', 'slp']]
y = df.iloc[:,13]

svm_model = svm.SVC(kernel='linear',  probability=True)
svm_model.fit(X, y)

filename = 'linear_svm_model.sav'
pickle.dump(svm_model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))

prediction = loaded_model.predict([[4, 7, 2]]) 
print(prediction)

if prediction[0] == 0:
    print("Less chance of heart attack")
else:
    print("More chance of heart attack")