# dependency
tensorflow 1.9.0
```
pip install tensorflow-gpu==1.9.0
```

# Run

```
python optimizers_explore.py
```

# Note
This code is based on the lecture code [here](https://gitlab.oit.duke.edu/dec18/intro_to_deep_learning/blob/master/lectures/02_example.ipynb).

We use a MLP with two hidden layers, each of which has 500 and 300 hidden neurons.
For a full reference of reasonable structures in MLP, you may refer the gallery [here](http://yann.lecun.com/exdb/mnist/).

# results
Test accuracy vs. learning rate
(2000 iterations, momentum is 0.9)

| learning rate  |  SGD | Momentum  |  Adam |
|---|---|---|---|
| 0.00001  | 0.1029  |  0.3378 | 0.8771  |
| 0.0001  |  0.313 |  0.8153 | 0.9437  |
| 0.001  | 0.8124  | 0.9145  |  0.9628 | 
| 0.01 | 0.9169  |  0.9624 |  0.9209 | 
| 0.1 |  0.958 |  0.8732 |  0.0892 |  
| 1.0 |  0.098 |  0.098 | 0.1032  |

Refer to this [interesting NIPS oral paper](https://papers.nips.cc/paper/7003-the-marginal-value-of-adaptive-gradient-methods-in-machine-learning.pdf) on a comprehensive study of different optimizers.
One interesting conclusion is adaptive optimizers may not be able to find a optimum for better generalization, but they are more robust and need less tuning.

[This tutorial](http://ruder.io/optimizing-gradient-descent/) also gives a good summary.

Test accuracy vs. learning rate and momentum
(2000 iterations)

| momentum  |  0.6  |  0.8 | 0.9  |  0.95 | 0.975 | 0.99
|---|---|---|---|---|---|---|
| lr=0.001  | 0.8761  |  0.9017 |  0.9145 |  0.9353  |  0.9448 | 0.9557|
| lr=0.01  |  0.9374 |  0.9474 |  0.9624 |  0.9618  | 0.9632  | 0.8814|
| lr=0.1  | 0.9601  | 0.9549  | 0.8732  |  0.101  |  0.0974 | 0.101 |

In general, you should decrease learning rate if you increase momentum (that is, if your ball has more momentum to roll down the hill, you'd better slow it down by a smaller learning rate).
[Caffe tutorial](http://caffe.berkeleyvision.org/tutorial/solver.html) introduces well how learning rate should adapt to momentum.

You might also take a look at the relationship between batch size, learning rate and momentum [in this paper](https://openreview.net/forum?id=B1Yy1BxCZ). 

Be aware that different deep learning frameworks have different [implementations](https://pytorch.org/docs/stable/optim.html#torch.optim.SGD) of momenturm SGD.
