!pip install pandas numpy matplotlib scikit-learn tensorflow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics
import tensorflow.keras.backend as K

# Custom R2 Score Metric
def r2_score(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/11990560/demos/main/test_df.csv')

# Drop outliers and clean data
outliers = df[df['psf'] > (df['psf'].mean() * 3.0)]
df = df.drop(outliers.index)

# training, testing, and validation splits
df_validate = df.sample(frac=0.1)
df_validate.to_csv('df_validate.csv')
X_val = df_validate.drop(['price', 'psf', 'utilities_included'], axis=1)
y_val = df_validate['price']
ypsf_val = df_validate['psf']
df_test_train = df.drop(df_validate.index)
X = df_test_train.drop(['price', 'psf', 'utilities_included'], axis=1)
y = df_test_train['price']
ypsf = df_test_train['psf']
y_val = ypsf_val

# Split the data
rs = np.random.randint(100)
X_train, X_test, y_train, y_test = train_test_split(X, ypsf, test_size=0.5, random_state=rs)

# Building the model
model = Sequential()
af = 'relu'
model.add(Dense(256, activation=af))
model.add(Dense(256, activation=af))
model.add(Dense(256, activation=af))
model.add(Dense(256, activation=af))
model.add(Dense(256, activation=af))
model.add(Dense(256, activation=af))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam',
              loss='mse',
              metrics=[metrics.MeanAbsoluteError(),
                       metrics.MeanAbsolutePercentageError(),
                       r2_score])

# Train the model
h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30)

# Evaluate the model
y_pred = model.predict(X_val)
y_pred = np.concatenate(y_pred, axis=0)
error = y_pred - y_val
errorp = (y_pred - y_val) / y_val

# Print evaluation metrics
print(round(abs(error).mean(), 2), round(error.std(), 2))
print(round(abs(errorp).mean(), 2), round(errorp.std(), 2), round(errorp.mean(), 2))
print(model.evaluate(X_val, y_val, verbose=2))

# Plot errors and results
plt.bar(y_val.index, errorp, color='blue', width=10.0)
plt.title('Errors - %')
plt.show()
plt.scatter(y_val, y_pred)
plt.title('Prediction vs Real in $')
plt.show()

# Plot training history
plt.plot(h.history['r2_score'])
plt.plot(h.history['val_r2_score'])
plt.title('Model R2 Score')
plt.ylabel('R2 Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot loss history
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
