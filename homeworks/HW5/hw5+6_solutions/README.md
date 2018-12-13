
# (i) MLP with word embedding
Dataset: `movie-pang02.txt`
 
dimensions: 300, 60. 

Final accuracy: ~ 0.666

# (ii) RNN with word embedding
Dataset: `movie-simple.txt`

Adam learning rate is 0.0002. 

RNN dimension is 50. 

Final accuracy: ~ 0.9291784702549575

# (iii) MLP with one-hot coding
Dataset: `movie-pang02.txt`
 
dimensions: 300, 60. 

Final accuracy: 0.4719999969005585

# (iv) RNN with one-hot coding
Dataset: `movie-simple.txt`

Adam learning rate is 0.0002. 

RNN dimension is 50. 

Final accuracy: 0.9065155807365439

# (v)
Word embeddings are representations with semantics. They are usually learned from tons of data in a unsupervised way, which gives much more semantic info for learning. Moreover, models are smaller with word embeddings, making them easier to optimize. A practical benefit of word embeddings is memory efficient and computation efficient; one-hot models easily run out of memory.

Word embeddings give higher accuracy when your dataset is more complex, such as `movie-pang02.txt`. You may get trivial difference when using `movie-simple.txt`.
