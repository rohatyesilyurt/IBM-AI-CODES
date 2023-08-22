import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd


# verilerimizi alalım. kullanacağımız veriler sklearn kütüphanesinin ünlü data setlerinden biri olan "iris" data seti olacak.
iris = datasets.load_iris()

# Sepal Length, Sepal Width, Petal Length ve Petal Width adında 4 sütundan oluşan 150x4 boyutunda bir veri.
# bu verilerden Sepal Width ve Petal Width sütunlarını kullanacağız. 
 
x = iris.data[:, [1,3]] # "x" değerlerimizde 1. ve 3. sütunları seçiyoruz.
y = iris.target # hedef verilerimizi seçiyoruz.

# Eğitim prosedürü "logistic regression" ile neredeyse aynıdır.

lr = LogisticRegression(random_state=0).fit(x, y)

# verilerin hangi olasıklara sahip olduklarına bakalım.
probability = lr.predict_proba(x)

# şimdi softmax prediction kullanabiliriz
#  bu kod satırı, her satırda en yüksek olasılığa sahip sınıfın indeksini bulmaya yarar.
softmax_prediction=np.argmax(probability,axis=1)

# ne kadar doğru bir tahminde bulunduğumuza bakalım
for i in range(len(y)):
    if y[i] == softmax_prediction[i]:
        print(f"{softmax_prediction[i]} - {y[i]}")
    else:
        print(f"{softmax_prediction[i]} - {y[i]} YANLIŞ")
accuracy = accuracy_score(y, softmax_prediction)
print("Accuracy:", accuracy)
