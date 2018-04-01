import math
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np

L, N = 0.0005, 0.1

(train_images, train_labels, test_images, test_labels) = pickle.load(open('./mnist_data/data.pkl', 'rb'),
                                                                     encoding='latin1')

train_images, test_images = train_images.astype(np.float64) / 256.0, test_images.astype(np.float64) / 256.0

w1, w2, w3 = np.zeros((784, 300)), np.zeros((300, 100)), np.zeros((100, 10))
for x in range(784):
    for y in range(300):
        w1[x][y] = random.uniform(-1 * np.sqrt(6.0 / (784 + 300)), 1 * np.sqrt(6.0 / (784 + 300)))
for x in range(300):
    for y in range(100):
        w2[x][y] = random.uniform(-1 * np.sqrt(6.0 / (300 + 100)), 1 * np.sqrt(6.0 / (100 + 300)))
for x in range(100):
    for y in range(10):
        w3[x][y] = random.uniform(-1 * np.sqrt(6.0 / (100 + 10)), 1 * np.sqrt(6.0 / (100 + 10)))

loss, accuracy = [], []
train_images_100, train_labels_100 = np.split(train_images, 100, axis=0), np.split(train_labels, 100, axis=0)
b3, b2, b1 = 0, 0, 0

for epoch in range(0, 3961):
    if epoch == int(0.5 * 3961):
        N = 0.01

    random_num = random.randint(0, 99)
    current_train_images, current_train_labels = train_images_100[random_num], train_labels_100[random_num]

    # Forward propagation
    a1 = current_train_images.dot(w1) + b1
    z1 = np.where(a1 > 0.0, a1, 0.0)
    a2 = z1.dot(w2) + b2
    z2 = np.where(a2 > 0.0, a2, 0.0)
    a3 = z2.dot(w3) + b3

    error_value, count = 0, 0
    xita_3 = None

    for row in a3:
        # print(row)
        temp = np.exp(row) / np.sum(np.exp(row), axis=0)  # Softmax changing
        temp[np.isnan(temp)] = 0.9
        aa = np.where(temp == np.max(temp))

        correct = current_train_labels[count]  # correct number index
        temp[correct] = temp[correct] if temp[correct] != 0 else 0.1
        error_value = error_value - math.log(temp[correct])  # loss function

        temp[correct] = temp[correct] - 1

        xita_3 = temp if xita_3 is None else np.vstack([xita_3, temp])

        count = count + 1

    xita_3 = xita_3 / 3.5
    w3_1 = (w3 - N * z2.transpose().dot(xita_3) - N * L * w3)
    b3_1 = (b3 - N * xita_3.mean())

    xita_2 = xita_3.dot(w3.transpose()) * np.where(z2 > 0.0, 1, 0) / 3.5
    w2_1 = w2 - N * z1.transpose().dot(xita_2) - N * L * w2
    b2_1 = b2 - N * xita_2.mean()

    xita_1 = xita_2.dot(w2.transpose()) * np.where(z1 > 0.0, 1, 0) / 3.5
    w1_1 = w1 - N * current_train_images.transpose().dot(xita_1) - N * L * w1
    b1_1 = b1 - N * xita_1.mean()

    w3, w2, w1, b3, b2, b1 = w3_1, w2_1, w1_1, b3_1, b2_1, b1_1

    # testing:
    if epoch % 40 == 0:
        ta1 = test_images.dot(w1) + b1
        tz1 = np.where(ta1 > 0.0, ta1, 0.0)
        ta2 = tz1.dot(w2) + b2
        tz2 = np.where(ta2 > 0.0, ta2, 0.0)
        ta3 = tz2.dot(w3) + b3
        count, correct_num = 0, 0
        for row in ta3:
            temp = np.exp(row) / np.sum(np.exp(row), axis=0)  # Softmax changing
            temp[np.isnan(temp)] = 1.0
            aa = np.where(temp == np.max(temp))

            correct = test_labels[count]  # correct number index

            if aa[0][0] == correct:
                correct_num = correct_num + 1

            count = count + 1
        print("loss:", float('%.4f' % (error_value / 100.0 + L / 2.0 * np.sum(w3 ** 2))))
        print("accuracy:", float(correct_num) / 1000.0)
        loss.append(float('%.4f' % (error_value / 100.0 + L / 2.0 * np.sum(w3 ** 2))))
        accuracy.append((float(correct_num) / 1000.0))

print("loss:", loss)
print("accuracy:", accuracy)

plt.figure("BPNN")
plt.tight_layout(True)
plt.subplot(211)
plt.plot(accuracy, 'k')
plt.ylabel("accuracy")
plt.grid(True)
plt.tight_layout(True)
plt.subplot(212)
plt.plot(loss, 'r')
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid(True)
plt.tight_layout(True)
plt.savefig('figure.pdf', dbi=300)
plt.show()
