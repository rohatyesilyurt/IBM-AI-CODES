import numpy as np 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# verileri okuyun
my_data = pd.read_csv("drug200.csv", delimiter=",")

# print(my_data.shape) # veri sayısını gösterir 

x = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

# Cinsiyet gibi kategorik değişkenleri sayısal değişkenlere dönüştürmek için LabelEncoder kullanarak bu özellikleri sayısal değerlere dönüştürebiliriz.
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
x[:,1] = le_sex.transform(x[:,1]) 

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
x[:,2] = le_BP.transform(x[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
x[:,3] = le_Chol.transform(x[:,3]) 

y = my_data["Drug"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(x_train,y_train)

#tahmin
predTree = drugTree.predict(x_test)

print(predTree[0:5])
print(y_test.values[0:5])

#accuracy test
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

#data visualisation
tree.plot_tree(drugTree)
plt.show()