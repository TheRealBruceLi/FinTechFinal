# Bruce Li, Evan Yukevich, and Preston Vander Vos
# Carnegie Mellon University
# 70-339 FinTech
# Final Project

# Source: tensorflow.org

import tensorflow as tf
from data import *

def categorizeData(data, labels):
    # Converts data into seperate lists to be used later
    prices = []
    for price in trainLabels.tolist():
        prices.append(float(price[0]))
    (data1, data2, data3, data4, data5, data6) = ([],[],[],[],[],[])
    for cate in trainData.tolist():
        data1.append(float(cate[0]))
        data2.append(float(cate[1]))
        data3.append(float(cate[2]))
        data4.append(float(cate[3]))
        data5.append(float(cate[4]))
        data6.append(float(cate[5]))
    return (prices, data1, data2, data3, data4, data5, data6)

(DATA, LABELS) = preprocessData("data/")
(DATAN, LABELSN) = normalizeData(DATA, LABELS)
(trainData,trainLabels,validateData,validateLabels) = splitData(DATAN,LABELSN)
(prices,data1,data2,data3,data4,data5,data6)=categorizeData(trainData,trainLabels)
(valPrices, valData1, valData2, valData3, valData4, valData5,
valData6) = categorizeData(validateData, validateLabels)

tf.reset_default_graph()
input1 = tf.placeholder(dtype=tf.float32, shape=None)
input2 = tf.placeholder(dtype=tf.float32, shape=None)
input3 = tf.placeholder(dtype=tf.float32, shape=None)
input4 = tf.placeholder(dtype=tf.float32, shape=None)
input5 = tf.placeholder(dtype=tf.float32, shape=None)
input6 = tf.placeholder(dtype=tf.float32, shape=None)
output_data = tf.placeholder(dtype=tf.float32, shape=None)
coeff1 = tf.Variable(0.1, dtype=tf.float32)
coeff2 = tf.Variable(0.1, dtype=tf.float32)
coeff3 = tf.Variable(0.1, dtype=tf.float32)
coeff4 = tf.Variable(0.1, dtype=tf.float32)
coeff5 = tf.Variable(0.1, dtype=tf.float32)
coeff6 = tf.Variable(0.1, dtype=tf.float32)
intercept = tf.Variable(1, dtype=tf.float32)
# y = m1x1 + m2x2 + m3x3 + m4x4 + m5x5 + m6x6 + b
model_operation = ((coeff1*input1) + (coeff2*input2) + (coeff3*input3) +
(coeff4*input4) + (coeff5*input5) + (coeff6*input6) + intercept)
error = model_operation - output_data
squared_error = tf.square(error)
loss = tf.reduce_mean(squared_error)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
x1 = data1
x2 = data2
x3 = data3
x4 = data4
x5 = data5
x6 = data6
y_values = prices

epochs = 15001
with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        sess.run(train, feed_dict={input1:x1, input2:x2, input3:x3, input4:x4,
        input5:x5, input6:x6, output_data:y_values})
        if i % 1000 == 0:
            print("training loss:",loss.eval(feed_dict={input1:x1, input2:x2,
            input3:x3, input4:x4, input5:x5, input6:x6, output_data:y_values})
            *len(prices))
            answer = (sess.run([coeff1, coeff2, coeff3, coeff4, coeff5,
            coeff6, intercept]))
            squared_error_sum = 0
            for i in range(len(valPrices)):
                guess = ((answer[0]*valData1[i])+(answer[1]*valData2[i])+
                (answer[2]*valData3[i])+(answer[3]*valData4[i])+
                (answer[4]*valData5[i])+(answer[5]*valData6[i])+answer[6])
                error = valPrices[i] - guess
                squared_error_sum += error**2
            print("validation loss:",squared_error_sum)
        if i == epochs-1:
            answer = (sess.run([coeff1, coeff2, coeff3, coeff4, coeff5,
            coeff6, intercept]))
            for i in range(6):
                print("coefficient",str(i)+":",answer[i])
            print("intercept:",answer[6])
