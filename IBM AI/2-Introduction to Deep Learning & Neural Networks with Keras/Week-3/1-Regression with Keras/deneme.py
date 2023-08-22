from keras.models import load_model
import numpy as np

loaded_model = load_model('modelim.h5')

new_house_features = np.array([[332.5, 142.5, 0.0, 228.0, 0.0, 932.0, 594.0, 365]])
new_house_norm = (new_house_features - new_house_features.mean()) / new_house_features.std()
# Tahmin yapma
prediction1 = loaded_model.predict(new_house_norm)

print(prediction1)