# setup env
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# load data
cells = pd.read_csv(r'../../four/CellDNA.csv', header = None)

x = cells.copy().loc[:, 1:13]

y = cells.copy().loc[:, 13]

del cells

# set 1 to interesting and zero to not interesting
interesting_index = y == 0

y.loc[interesting_index] = 1

y.loc[-interesting_index] = 0

del interesting_index

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3)

del x, y

# standardize X
std_fit = StandardScaler().fit(x_train)

x_train = std_fit.transform(x_train)

x_test = std_fit.transform(x_test)

# model --------------------------------------------

# build the model
nn = Sequential()

# input layer
nn.add(Dense(13, activation = 'relu', input_shape = (13, )))

# hidden layer
nn.add(Dense(8, activation = 'relu'))

# output layer
nn.add(Dense(1, activation = 'sigmoid'))

nn.output_shape

nn.summary()

nn.get_weights()

# train the model
nn.compile(loss = 'binary_crossentropy',
           optimizer = 'adam',
           metrics = ['accuracy'])

nn.fit(x_train, y_train, epochs = 100, batch_size = 128, verbose = 1, shuffle = True)

# predict on hold test set
y_hat = nn.predict(x_test)

# print loss and accuracy
nn.evaluate(x_test, y_test, verbose = 1)

# convert from probabilities to classes
y_hat_binary = np.ravel(y_hat)

interesting_index = y_hat_binary >= .5

y_hat_binary[interesting_index] = 1

y_hat_binary[np.logical_not(interesting_index)] = 0

# test-set confusion matrices
print(confusion_matrix(y_test, y_hat))

# add origin point because otherwise ROC will just be flat line since correctly classified all observations
def add_origin(fpr, tpr):
    fpr = np.r_[0, fpr]
    tpr = np.r_[0, tpr]

    return fpr, tpr

# Interesting roc curve
fpr, tpr, _ = roc_curve(y_test, y_hat)

fpr, tpr = add_origin(fpr, tpr)

plt.plot(fpr, tpr, label = 'Interesting', color = 'blue')

# Non-Interesting roc curve
y_hat_non_interesting = 1 - y_hat

fpr, tpr, _ = roc_curve(y_test, y_hat_non_interesting, pos_label = 0)

fpr, tpr = add_origin(fpr, tpr)

plt.plot(fpr, tpr, label = 'Non-Interesting', color = 'red')
plt.legend()
plt.title('Binary-Class, ROC Curve')
plt.show()
