import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

# verileri okuyun
df = pd.read_csv("FuelConsumptionCo2.csv")

# veri kümesine bir göz atın
# print(df.head())

# verileri özetleyin
#print(df.describe())

# Regresyon için kullanmak istediğimiz bazı özellikleri seçelim.
cdf = df[['ENGINESIZE','CYLINDERS','CO2EMISSIONS']]

# verilerin seçilen sütunlarını gösterir 
# print(cdf.head(10))

# aralarındaki ilişkinin ne kadar doğrusal olduğunu görmek için bu değerler ile Emisyon değerlerini grafikleştirin
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color ='blue')
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.show()

# elimizdeki veriyi %80-%20 oranına en yakın olacak şekilde bölüyoruz. 
# Kesin olarak ayırma yöntemi de var ama rastgele bölme, zaman içinde daha genel bir değerlendirme yapmamıza olanak tanır.

msk = np.random.rand(len(df)) < 0.8 
train = cdf[msk]
test = cdf[~msk]

# veri̇ modelleme
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])  # np.asanyarray, Python listelerini veya benzeri dizileri NumPy dizilerine dönüştürmek için kullanılır.
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Katsayılar
print('Coefficients(katsayılar): ', regr.coef_) # denklem için gerekli olan katsayılar
print('Intercept(sabit): ', regr.intercept_) # denklem için gerekli olan sabit 

# bulduğumuz doğruyu verilerin üzerine çizebiliriz:
# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
# plt.plot((train_x), (regr.coef_[0][0]*train_x + regr.intercept_[0]), '-r')  
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])  # np.asanyarray, Python listelerini veya benzeri dizileri NumPy dizilerine dönüştürmek için kullanılır.
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_prediction = regr.predict(test_x)

print(f"Mean absolute error:{np.mean(np.absolute(y_prediction - test_y)):.2f}")
print(f"Residual sum of squares (MSE): {np.mean((y_prediction - test_y) ** 2):.2f}")
print(f"R2-score: {r2_score(test_y , y_prediction):.2f}")

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.scatter(test_x, test_y, color='red')
plt.scatter(test_x, y_prediction, color='blue')
plt.plot(test_x, y_prediction, color='blue')
plt.show()