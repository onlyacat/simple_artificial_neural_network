import math
import pickle
import random

import matplotlib
import numpy as np

matplotlib.use('Agg')

# Definition of functions and parameters
# for example
EPOCH = 1000
LAMUDA = 0.0005
NIGA = 0.1
NUM_PER_ONE = 100

# Read all data from .pkl
(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),
                                                                     encoding='latin1')

### 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256

train_images = train_images.astype(np.float64)
train_images = train_images / 256.0

### 2. Weight initialization: Xavier

w1 = np.zeros((784, 300))
w2 = np.zeros((300, 100))
w3 = np.zeros((100, 10))
for x in range(784):
    for y in range(300):
        w1[x][y] = random.uniform(-1 * np.sqrt(6.0 / (784 + 300)), 1 * np.sqrt(6.0 / (784 + 300)))
for x in range(300):
    for y in range(100):
        w2[x][y] = random.uniform(-1 * np.sqrt(6.0 / (300 + 100)), 1 * np.sqrt(6.0 / (100 + 300)))
for x in range(100):
    for y in range(10):
        w3[x][y] = random.uniform(-1 * np.sqrt(6.0 / (100 + 10)), 1 * np.sqrt(6.0 / (100 + 10)))

# np.random.shuffle(w1)
# np.random.shuffle(w2)
# np.random.shuffle(w3)

### 3. training of neural network

loss = np.zeros((EPOCH))
accuracy = np.zeros((EPOCH))
train_images_100 = np.split(train_images, NUM_PER_ONE, axis=0)
train_labels_100 = np.split(train_labels, NUM_PER_ONE, axis=0)
b3, b2, b1 = 0, 0, 0

for epoch in range(0, EPOCH):
    if epoch == int(0.5 * EPOCH):
        NIGA = 0.01

    random_num = random.randint(0, 99)
    current_train_images = train_images_100[random_num]
    current_train_labels = train_labels_100[random_num]

    # Forward propagation
    a1 = current_train_images.dot(w1) + b1
    z1 = np.where(a1 > 0.0, a1, 0.0)
    a2 = z1.dot(w2) + b2
    z2 = np.where(a2 > 0.0, a2, 0.0)
    a3 = z2.dot(w3) + b3

    ans = np.zeros((NUM_PER_ONE, 1))
    error_value = 0
    count = 0
    xita_3 = None

    for row in a3:
        # print(row)
        temp = np.exp(row) / np.sum(np.exp(row), axis=0)  # Softmax changing
        aa = np.where(temp == np.max(temp))
        ans[count] = aa[0][0]

        correct = current_train_labels[count]  # correct number index
        '''if ans[count] == correct:
            error_value = error_value - math.log(np.max(temp))
        '''
        error_value = error_value - math.log(temp[correct])  # loss function

        # if xita_3 :
        #     xita_3 =
        for x in range(0, 10):
            if x == correct:
                temp[x] = temp[x] - 1

        if xita_3 is None:
            xita_3 = temp
        else:
            xita_3 = np.vstack([xita_3, temp])

        count = count + 1

    loss[epoch] = error_value / NUM_PER_ONE

    xita_3 = xita_3/30
    w3_1 = (w3 - NIGA * z2.transpose().dot(xita_3) - NIGA * LAMUDA * w3)
    b3_1 = (b3 - NIGA * xita_3.mean())

    xita_2 = xita_3.dot(w3.transpose()) * np.where(z2 > 0.0, 1, 0)/30
    w2_1 = w2 - NIGA * z1.transpose().dot(xita_2) - NIGA * LAMUDA * w2
    b2_1 = b2 - NIGA * xita_2.mean()

    xita_1 = xita_2.dot(w2.transpose()) * np.where(z1 > 0.0, 1, 0)/30
    w1_1 = w1 - NIGA * current_train_images.transpose().dot(xita_1) - NIGA * LAMUDA * w1
    b1_1 = b1 - NIGA * xita_1.mean()

    w3, w2, w1, b3, b2, b1 = w3_1, w2_1, w1_1, b3_1, b2_1, b1_1

    # testing:
    if epoch % 100 == 0 or epoch == EPOCH - 1:
        ta1 = test_images.dot(w1) + b1
        tz1 = np.where(ta1 > 0.0, ta1, 0.0)
        ta2 = tz1.dot(w2) + b2
        tz2 = np.where(ta2 > 0.0, ta2, 0.0)
        ta3 = tz2.dot(w3) + b3
        count = 0
        correct_num = 0
        for row in ta3:
            # print(row)
            temp = np.exp(row) / np.sum(np.exp(row), axis=0)  # Softmax changing
            temp[np.isnan(temp)] = 1.0
            aa = np.where(temp == np.max(temp))

            correct = test_labels[count]  # correct number index

            if aa[0][0] == correct:
                correct_num = correct_num + 1

            count = count + 1

        print(epoch, " ", loss[epoch], " ", correct_num)

# Back propagation

# Gradient update

# Testing for accuracy


### 4. Plot
# for example
# plt.figure(figsize=(12,5))
# ax1 = plt.subplot(111)
# ax1.plot(......)
# plt.xlabel(......)
# plt.ylabel(......)
# plt.grid()
# plt.tight_layout()
# plt.savefig('figure.pdf', dbi=300)
