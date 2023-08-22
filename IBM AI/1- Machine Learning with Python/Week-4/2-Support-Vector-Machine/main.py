import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm

cell_df = pd.read_csv("cell_samples.csv")

# Önce sütun veri türlerine bakalım:
print(cell_df.head())
print(cell_df.dtypes)

# Görünüşe göre "BareNuc" sütunu sayısal olmayan bazı değerler içeriyor. Bu satırları çıkarabiliriz:
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

x = np.asarray(cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']])
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)

# #clf = classifier,  SVC = SUPPORT VECTOR CLASSIFIER, svm = SUPPORT VECTOR MACHINE
#  kernel değerini  "['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']" gibi değerler yaparak farklı modeller kullanabiliriz.
clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(y_test[0:10])
print(y_pred[0:10])

counter_true = 0
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print(f"{y_pred[i]}, {y_test[i]} : Doğru")
        counter_true = counter_true + 1 
    else:
        print(f"{y_pred[i]}, {y_test[i]}")

print(f"Veri sayısı: {len(y_test)} Doğru: {counter_true} Yanlış: {len(y_test) - counter_true}")
