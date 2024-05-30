# there are built-in models for LSTM and GRU
# if needed, get_weight() shall be considered

import math
import numpy as np

def relu(X):
    y=0
    if X>0:
        y=X

    return y

def sigmoid(x):
    return 1/(1+math.exp(-x))

# math.tanh(x)
def tanh(x):
    return math.tanh(x)

def net(w1,w2,b,x1,x2):
    return w1*x1+w2*x2+b

# or, set vi = np.array([w1,w2,b])
# vx = np.array([x1,x2,1])
# d = y_true
# y = forward propagation

def backprop(w1,w2,b,x1,x2,y,d,a):
    w1=w1+a*(d-y)*x1
    w2=w2+a*(d-y)*x2
    b=b+a*(d-y)
    print(w1)
    print(w2)
    print(b)
    return w1, w2, b    
