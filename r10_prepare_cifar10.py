from keras.datasets import cifar10
from keras.utils import to_categorical

import numpy
from numpy import savetxt


print("begin import the cifar-10 from keras.datasets")

num_class = 10
batchsize = 128
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float16') / 255.
x_test_raw = x_test
x_test = x_test.astype('float16') / 255.
y_train = to_categorical(y_train, num_class)
y_test = to_categorical(y_test, num_class)

export = x_test.reshape(10000, 3072)
x_train = x_train.reshape(50000,32,32,3)

print("data reshape complete: float16 with test 10000 samples")

with open("data.csv", "ab") as f:
    for i in range(len(export)):
        numpy.savetxt(f, export[i], delimiter=',')
print("finished data writting data.csv")

with open("label.csv", "ab") as f:
    for i in range(len(y_test)):
        numpy.savetxt(f, y_test[i], delimiter=',')
print("finished label writting label.csv")


print("data.csv: 3207x10000 rows, with each row contains one pixel in float16")
print("label.csv: 10x10000 rows, with each row contains one of ten FE layer output")