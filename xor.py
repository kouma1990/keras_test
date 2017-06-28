import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

np.random.seed(0)

model = Sequential([
    Dense(input_dim=2, units=2),
    Activation("sigmoid")
])

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer=SGD(lr=0.1))

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0] , [1], [1], [0]])

model.fit(X, Y, epochs=4000, batch_size=4)

classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

print(prob)