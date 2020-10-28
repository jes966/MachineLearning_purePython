import numpy as np

def error_val(x,t,W,b):
    y=np.dot(x,W)+b

    return (np.sum((t-y)**2))/(len(x))

def predict(x,W,b):
    y=np.dot(x,W)+b

    return y

def loss_func(x,t,W,b):
    y=np.dot(x,W)+b

    return (np.sum((t-y)**2)/len(x))