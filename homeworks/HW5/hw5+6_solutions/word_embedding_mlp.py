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

# Placeholders for input
X = tf.placeholder(tf.float32, [None, 300])
y = tf.placeholder(tf.float32, [None, 1])

# Three-layer MLP
h1 = tf.layers.dense(X, 300, tf.nn.relu)
h2 = tf.layers.dense(h1, 60, tf.nn.relu)
logits = tf.layers.dense(h2, 1)
probabilities = tf.sigmoid(logits)

# Loss and metrics
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(logits)), y), tf.float32))

# Training
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Initialization of variables
initialize_all = tf.global_variables_initializer()


# Apply the function to each line in the file.
with open("../../lectures/Data/movie-pang02.txt", "r",encoding=enc) as f:
    dataset = [convert_line_to_example(l) for l in f.readlines()]
import random
random.shuffle(dataset)
batch_size = 100
total_batches = len(dataset) // batch_size
train_batches = 3 * total_batches // 4
train, test = dataset[:train_batches*batch_size], dataset[train_batches*batch_size:]
sess = tf.InteractiveSession()
sess.run(initialize_all)
for epoch in range(250):
    for batch in range(train_batches):
        data = train[batch*batch_size:(batch+1)*batch_size]
        reviews = [sample['x'] for sample in data]
        labels  = [sample['y'] for sample in data]
        labels = np.array(labels).reshape([-1, 1])
        _, l, acc = sess.run([train_step, loss, accuracy], feed_dict={X: reviews, y: labels})
    if epoch % 10 == 0:
        print("Epoch", epoch, "Loss", l, "Acc", acc)
    random.shuffle(train)

# Evaluate on test set
test_reviews = [sample['x'] for sample in test]
test_labels  = [sample['y'] for sample in test]
test_labels = np.array(test_labels).reshape([-1, 1])
acc = sess.run(accuracy, feed_dict={X: test_reviews, y: test_labels})
print("Final accuracy:", acc)
sess.close()

