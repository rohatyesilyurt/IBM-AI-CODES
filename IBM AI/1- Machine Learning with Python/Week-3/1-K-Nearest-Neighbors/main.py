import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Veriyi oku ve göz at
df = pd.read_csv('teleCust1000t.csv')
# print(df.head())

# Veri setimizde her sınıftan kaç tane olduğunu görelim
# print(df['custcat'].value_counts())


#To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
x = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values  #.astype(float)
y = df['custcat'].values

# verilerimizi train/test olarak ayırıyoruz ve test verisi boyutunu %20 olarak belirliyoruz. bu sayede out-of-sample accuracy değerimiz artacak
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
# print ('Train set:', x_train.shape, y_train.shape) # eğitim setinin boyutunu verir. Örneğin, (100, 10) ifadesi, 100 örnek ve her bir örnek için 10 özelliğe sahip bir eğitim verisi olduğunu gösterir.      
# print ('Test set:', x_test.shape, y_test.shape) # test setinin boyutunu verir.

# verilerimizi, ortalamaları "0" ve varyans(dağılım ölçüsü) değerleri "1" şekilde ölçeklendirerek KNN modelinin performansını artırıyoruz.
x_train_norm = preprocessing.StandardScaler().fit(x_train).transform(x_train.astype(float)) 
# print(x_train_norm[0:5]) # ilk 5 değerin ölçeklendirildikten sonraki halini ve önceki halini görebiliriz.
# print(x_train[0:5])

#Modeli Eğit ve Tahmin Et  
k = 4 # şimdilik 4 değerini verdik
neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train_norm,y_train)

x_test_norm = preprocessing.StandardScaler().fit(x_test).transform(x_test.astype(float))

y_pred = neigh.predict(x_test_norm)
print(y_pred[0:10])
print(y_test[0:10])   

print("Train set Accuracy 4 k's: ", metrics.accuracy_score(y_train, neigh.predict(x_train_norm)))
print("Test set Accuracy 4 k's:", metrics.accuracy_score(y_test, y_pred))


# We can calculate the accuracy of KNN for different values of k.

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(x_train_norm,y_train)
    y_pred=neigh.predict(x_test_norm)

    mean_acc[n-1] = metrics.accuracy_score(y_test, y_pred)
    std_acc[n-1]=np.std(y_pred==y_test)/np.sqrt(y_pred.shape[0])


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)