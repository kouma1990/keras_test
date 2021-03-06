import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation
#from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn import datasets
from sklearn.model_selection import train_test_split


N = 300
X, y = datasets.make_moons(N, noise=0.3)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8)


model = Sequential()

model.add(Dense(100, input_dim=2, init="he_normal"))
model.add(Activation("relu"))
model.add(Dense(100, input_dim=2, init="he_normal"))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999), metrics=["accuracy"])

early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1)
hist = model.fit(x_train, y_train, epochs=500, batch_size=20, validation_data=(x_train, y_train), callbacks=[early_stopping])

loss_and_metrics = model.evaluate(x_test, y_test)

print(loss_and_metrics)

# グラフ描画
val_acc = hist.history["val_acc"]

plt.rc("font", family="serif")
fig = plt.figure()
plt.plot(range(500), val_acc, label="acc", color="black")
plt.xlabel("epochs")
plt.show() 