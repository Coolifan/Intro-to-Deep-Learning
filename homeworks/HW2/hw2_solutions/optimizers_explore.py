import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Import data
validation_size=5000
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=validation_size)

tf.reset_default_graph()

# Model Inputs
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Define the graph
y=tf.layers.dense(x,units=500,activation=tf.nn.relu)
y=tf.layers.dense(y,units=300,activation=tf.nn.relu)
y_logits=tf.layers.dense(y,10,activation=None)

# Loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_logits))

# Evaluation
correct_prediction = tf.equal(tf.argmax(y_logits, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train_MLP(train_step_optimizer, iterations=4000):
    with tf.Session() as sess:
        # Initialize (or reset) all variables
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # Initialize arrays to track losses and validation accuracies
        valid_accs = []
        losses = []

        for i in range(iterations):
            # Validate every 250th batch
            if i % 250 == 0:
                validation_accuracy = 0
                valid_batch_size = 50
                steps = validation_size // valid_batch_size
                for v in range(steps):
                    batch = mnist.validation.next_batch(valid_batch_size)
                    validation_accuracy += (1.0 / steps) * accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
                print('step %d, validation accuracy %g' % (i, validation_accuracy))
                valid_accs.append(validation_accuracy)

            # Train
            batch = mnist.train.next_batch(20)
            loss, _ = sess.run([cross_entropy, train_step_optimizer], feed_dict={x: batch[0], y_: batch[1]})
            losses.append(loss)

        print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

    return valid_accs, losses

# hyperparameters you can play with
lr = 0.001
momentum=.9
iterations = 2000

train_step_SGD = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(cross_entropy)
# We can incoporate momentum by simply adding one line
train_step_momentum = tf.train.MomentumOptimizer(learning_rate=lr,momentum=momentum).minimize(cross_entropy)
# We can also try ADAM
train_step_adam = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

initialize_all= [tf.global_variables_initializer(),tf.local_variables_initializer()]
print("SGD:")
valid_accs_SGD, losses_SGD = train_MLP(train_step_SGD,iterations)
print("momentum:")
valid_accs_momentum, losses_momentum = train_MLP(train_step_momentum,iterations)
print("ADAM:")
valid_accs_adam, losses_adam = train_MLP(train_step_adam,iterations)



fig, ax = plt.subplots()

ax.plot(valid_accs_SGD)
ax.plot(valid_accs_momentum)
ax.plot(valid_accs_adam)

ax.set_ylabel('Validation Accuracy')
ax.set_xlabel('Evaluations')
ax.legend(['SGD', 'Momentum', 'Adam'], loc='lower right')
# saving as pdf
fig.savefig("validation_accuracy.pdf")