import pandas as pd
import numpy as np
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense


concrete_data = pd.read_csv("concrete_data.csv")
# print(concrete_data.head())
# print(concrete_data.shape)

#dataset içinde kayıp/bozuk veri olup olmadığını kontrol edelim
# print(concrete_data.describe())
# print(concrete_data.isnull().sum())

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

# verileri normalize ediyoruz.  ortalamayı çıkarıp standart sapmaya bölerek elde edilir.
predictors_norm = (predictors - predictors.mean()) / predictors.std()
print(predictors.head()) # normalize etmeden önce
print(predictors_norm.head()) #normalize ettikten sonra

n_cols = predictors_norm.shape[1] # "predictor" sayısını öğrenelim. "shape" fonskiyonu (1030,8) şeklinde bir çıktı verecek ve biz shape[1] yaparak "8" sayısına ulaşmaya çalışıyoruz
print(n_cols)

# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# build the model
model = regression_model()

# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

model.save('modelim.h5')

