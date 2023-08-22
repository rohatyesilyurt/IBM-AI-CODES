import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Veriyi oku.
churn_df = pd.read_csv("ChurnData.csv")

# Modelleme için istediğimiz bazı özellikleri seçelim.
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip','callcard', 'wireless','churn']]
# Ayrıca, skitlearn algoritmasının bir gerekliliği olduğu için hedef veri türünü tamsayı olarak değiştiriyoruz:
churn_df['churn'] = churn_df['churn'].astype('int')
# print(churn_df.head())
# print(churn_df.shape)

x = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
# print(x[0:5])

y = np.asarray(churn_df['churn'])
# print(y[0:5])

# Veriyi ölçeklendirelim.
# Veriyi ölçeklendirirken her özelliğin ortalama değerini 0'a ve standart sapmasını 1'e dönüştürmeyi amaçlar.
# Bu işlem, her özelliğin aynı ölçekte olmasını sağlamak ve makine öğrenimi algoritmalarının daha iyi sonuçlar vermesine yardımcı olmak için yapılır.
x = preprocessing.StandardScaler().fit(x).transform(x)
# print(x[0:5])

# Veri kümemizi eğitim ve test kümelerine ayırdık
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
# print ('Train set:', x_train.shape,  y_train.shape)
# print ('Test set:', x_test.shape,  y_test.shape)

# "C" parametresi, pozitif bir float değer olması gereken **düzenleme gücünün** tersini gösterir. Daha küçük değerler daha güçlü düzenlileştirmeyi belirtir.
LR = LogisticRegression(C=0.01, solver='liblinear').fit(x_train,y_train)

#tahmin
y_pred = LR.predict(x_test)

# **predict_proba** sınıfların etiketine göre sıralanmış tüm sınıflar için tahminler döndürür. Yani, ilk sütun sınıf 0'ın olasılığıdırikinci sütun sınıf 1'in olasılığıdır:
y_pred_prob = LR.predict_proba(x_test)
print(y_pred_prob)
