import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model

# verileri okuyun
df = pd.read_csv("FuelConsumptionCo2.csv")

# veri kümesine bir göz atın
# print(df.head())

# verileri özetleyin
#print(df.describe())

# Regresyon için kullanmak istediğimiz bazı özellikleri seçelim.
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

# verilerin seçilen sütunlarını gösterir 
# print(cdf.head(10))

# Emisyon değerlerini Motor boyutuna göre grafikle gösterelim:
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

# elimizdeki veriyi %80-%20 oranına en yakın olacak şekilde bölüyoruz. 
# Kesin olarak ayırma yöntemi de var ama rastgele bölme, zaman içinde daha genel bir değerlendirme yapmamıza olanak tanır.
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Veri̇ modelleme:
# Burada iyi olan şey,
# çoklu doğrusal regresyon modelinin basit doğrusal regresyon modelinin bir uzantısı olmasıdır.
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

# Katsayılar
print('Coefficients(katsayılar): ', regr.coef_) # denklem için gerekli olan katsayılar
print('Intercept(sabit): ', regr.intercept_) # denklem için gerekli olan sabit 

test_x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
y_prediction= regr.predict(test_x)

# Explained variance score: 1 mükemmel tahmindir.
print(f"Variance score: {regr.score(test_x, test_y):2f}")



