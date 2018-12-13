# Download word vectors
import urllib
from urllib.request import urlretrieve
import os
if not os.path.isfile('mini.h5'):
    print("Downloading Conceptnet Numberbatch word embeddings...")
    conceptnet_url = 'http://conceptnet.s3.amazonaws.com/precomputed-data/2016/numberbatch/17.06/mini.h5'
    urlretrieve(conceptnet_url, 'mini.h5')

import numpy as np
import h5py
with h5py.File('mini.h5', 'r') as f:
    all_words = [word.decode('utf-8') for word in f['mat']['axis1'][:]]
    all_embeddings = f['mat']['block0_values'][:]

english_words = [word[6:] for word in all_words if word.startswith('/c/en/')]
english_word_indices = [i for i, word in enumerate(all_words) if word.startswith('/c/en/')]
english_embedddings = all_embeddings[english_word_indices]

norms = np.linalg.norm(english_embedddings, axis=1)
normalized_embeddings = english_embedddings.astype('float32') / norms.astype('float32').reshape([-1, 1])

index = {word: i for i, word in enumerate(english_words)}

# load reviews
import string
remove_punct=str.maketrans('','',string.punctuation)

# This function converts a line of our data file into
# a tuple (x, y), where x is 300-dimensional representation
# of the words in a review, and y is its label.
def convert_line_to_example(line):
    # Pull out the first character: that's our label (0 or 1)
    y = int(line[0])
    # Split the line into words using Python's split() function
    words = line[2:].translate(remove_punct).lower().split()
    # Look up the embeddings of each word, ignoring words not
    # in our pretrained vocabulary.
    embeddings = [normalized_embeddings[index[w]] for w in words
                  if w in index]
    # Take the mean of the embeddings
    x = np.mean(np.vstack(embeddings), axis=0)
    return {'x': x, 'y': y, 'w':embeddings}

# Apply the function to each line in the file.
enc = 'utf-8' # This is necessary from within the singularity shell

# building mlp
import tensorflow as tf
tf.reset_default_graph()

# sizes
n_steps = None
n_inputs = 300
n_neurons = 50
batch_size = 1
# Build RNN
X= tf.placeholder(tf.float32, [batch_size, n_steps, n_inputs])
y= tf.placeholder(tf.float32, [batch_size, 1])
basic_cell = tf.contrib.rnn.BasicRNNCell(n_neurons,activation=tf.nn.tanh)
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)
last_cell_output=outputs[:,-1,:]
y_=tf.layers.dense(last_cell_output,1)
# Loss and metrics
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, labels=y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_)), y), tf.float32))
# Training
train_step = tf.train.AdamOptimizer(0.0002).minimize(loss)


# Initialization of variables
initialize_all = tf.global_variables_initializer()


# Apply the function to each line in the file.
with open("../../lectures/Data/movie-simple.txt", "r",encoding=enc) as f:
    dataset = [convert_line_to_example(l) for l in f.readlines()]
import random
random.shuffle(dataset)
total_batches = len(dataset) // batch_size
train_batches = 3 * total_batches // 4
train, test = dataset[:train_batches*batch_size], dataset[train_batches*batch_size:]

sess = tf.InteractiveSession()
sess.run(initialize_all)
l_ma=.74
acc_ma=.5
for epoch in range(2):
    for batch in range(train_batches):
        data = train[batch*batch_size:(batch+1)*batch_size]
        reviews = np.array([sample['w'] for sample in data]).reshape([batch_size,-1,300])
        labels  = np.array([sample['y'] for sample in data]).reshape([batch_size,1])
        labels = np.array(labels).reshape([-1, 1])
        _, l, acc = sess.run([train_step, loss, accuracy], feed_dict={X: reviews, y: labels})
        l_ma=.99*l_ma+(.01)*l
        acc_ma=.99*acc_ma+(.01)*acc
        if (batch+1) % 100 == 0:
            print("batch", batch, "Loss", l_ma, "Acc", acc_ma)
    if epoch % 1 == 0:
        print("Epoch", epoch, "Loss", l_ma, "Acc", acc_ma)
    random.shuffle(train)

# Evaluate on test set
test_acc=0
n=0
for sample in test:
    test_reviews = np.array([sample['w'] ]).reshape([1,-1,300])
    test_labels  = np.array([sample['y']]).reshape([1,1])
    test_labels = np.array(test_labels).reshape([-1, 1])
    test_acc += sess.run(accuracy, feed_dict={X: test_reviews, y: test_labels})
    n+=1
acc=test_acc/n 
print("Final accuracy:", acc)
sess.close() 
