from common_used import utility
from common_used import numerical_derivative
import numpy as np

def Prepare_data():
    x_data=np.array([1,2,3,4,5]).reshape(5,1)
    t_data=np.array([2,3,4,5,6]).reshape(5,1)

    return x_data, t_data

if __name__ == '__main__':
    x_data,t_data=Prepare_data()

    W=np.random.rand(1,1)
    b=np.random.rand(1)   #initialize Weight and bias

    learning_rate=1e-2 #if the result diverge, change the value to 1e-3 or 1e-6 etc..

    f=lambda x: utility.loss_func(x_data,t_data,W,b)

    for step in range(8001):
        W-=learning_rate*numerical_derivative.numerical_derivative(f,W)
        b-=learning_rate*numerical_derivative.numerical_derivative(f,b)


        if (step%400==0):
            print('step= ',step, 'error value= ',utility.error_val(x_data,t_data,W,b),'W = ',W, 'b = ',b)

    print(utility.predict(43,W,b))
