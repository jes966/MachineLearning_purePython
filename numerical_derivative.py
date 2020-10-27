import numpy as np

def numerical_derivative(f,x):
    delta_x=1e-4
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])

    while not it.finished:
        idx=it.multi_index
        tmp_value=x[idx]

        x[idx]=float(tmp_value)+delta_x
        fx1=f(x)

        x[idx]=tmp_value-delta_x
        fx2=f(x)

        grad[idx]=(fx1-fx2)/(2*delta_x)

        x[idx]=tmp_value

        it.iternext()

    return grad