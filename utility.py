import numpy as np

def error_val(x,t):
    y=np.dot(x,W)+b

    return (np.sum((t-y)**2))/(len(x))

def predict(x):
    y=np.dot(x.W)+b

    return y