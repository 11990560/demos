### clocks
from datetime import datetime
start = datetime.now()
print("script start: ", start)

import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import metrics

# df = pd.read_csv('/content/drive/MyDrive/rfa/test_df.csv')
df = pd.read_csv('https://raw.githubusercontent.com/11990560/demos/main/test_df.csv')

print(len(df))
outliers = df[df['psf'] > (df['psf'].mean() * 3.0)]
df = df.drop(outliers.index)
print(len(df))

print(df[['price','psf','sq_feet']].describe())

# training, testing and validation splits
df_validate = df.sample(frac=0.1)
df_validate.to_csv('df_validate.csv')
X_val = df_validate.drop(['price','psf','utilities_included'], axis=1)
y_val = df_validate['price']
ypsf_val = df_validate['psf']
df_test_train = df.drop(df_validate.index)
X = df_test_train.drop(['price','psf','utilities_included'], axis=1)
y = df_test_train['price']
ypsf = df_test_train['psf']
y_val = ypsf_val # <------------------------------------------------------------------- toggle psf version

rs = np.random.randint(100)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=rs)
X_train, X_test, y_train, y_test = train_test_split(X, ypsf, test_size=0.5, random_state=rs) # psf version

# # normalization sequence
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)

# print(X_train, X_test, y_train, y_test)


### start neural netting ###
nn_start = datetime.now()
print("nn start: ", nn_start)


## Creating the model
model = Sequential()
af = 'relu'
e = 20

model.add(Dense(256,activation=af,name='layer1')) ##<----- You don't have to specify input size.Just define the hidden layers
model.add(Dense(256,activation=af,name='layer2'))
model.add(Dense(256,activation=af,name='layer3'))
model.add(Dense(256,activation=af,name='layer4'))
model.add(Dense(256,activation=af,name='layer5'))
model.add(Dense(256,activation=af,name='layer6'))
model.add(Dense(256,activation=af,name='layer7'))
model.add(Dense(256,activation=af,name='layer8'))
model.add(Dense(256,activation=af,name='layer9'))
model.add(Dense(256,activation=af,name='layer10'))
model.add(Dense(256,activation=af,name='layer11'))
model.add(Dense(256,activation=af,name='layer12'))
model.add(Dense(256,activation=af,name='layer13'))
model.add(Dense(1))

## defining the optimiser and loss function
model.compile(optimizer='adam'
              ,loss='mse'
              ,metrics=[metrics.MeanAbsoluteError()
                    ,metrics.MeanAbsolutePercentageError()
                    ,metrics.R2Score()
                    ]
              )

## training the model
h = model.fit(x=X_train
          ,y=y_train
          ,validation_data=(X_test,y_test)
          ,epochs=e
          )

y_pred = model(X_val)


### nn clock
nn_end = datetime.now()
print("nn end: ", nn_end)
print("nn duration: ", nn_end - nn_start)


# results diagnostics
y_pred = np.concatenate(y_pred, axis=0 )
error = y_pred - y_val
print(round(abs(error).mean(),2), round(error.std(),2))
errorp = (y_pred - y_val) / y_val
print(round(abs(errorp).mean(),2), round(errorp.std(),2),round(errorp.mean(),2))
print(model.evaluate(X_val,y_val,verbose=2))

# print(model.summary())
mae = mean(abs(y_val - y_pred))
mape = 100 * mean(abs((y_val - y_pred) / y_val))

plt.bar(y_val.index, errorp, color='blue', width=10.0)
plt.title('errors - %')
plt.show()
plt.scatter(y_val,y_pred)
plt.title('prediction vs real in $')
plt.show()

# list all data in history
# print(h.history.keys())
# summarize history for accuracy
plt.plot(h.history['r2_score'])
plt.plot(h.history['val_r2_score'])
plt.title('model r2_score')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for mape
plt.plot(h.history['mean_absolute_percentage_error'])
plt.plot(h.history['val_mean_absolute_percentage_error'])
plt.title('model mape')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for mae
plt.plot(h.history['mean_absolute_error'])
plt.plot(h.history['val_mean_absolute_error'])
plt.title('model mae')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

### clocks
end = datetime.now()
print("script start: ", start)
print("script end: ", end)
print("total time: ", end - start)
print("neural net duration: ", nn_end - nn_start)
