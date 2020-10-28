#미분은 함수 f(x)가 있을 경우 x가 미세하게 움직였을 때 그 결과가 어떻게 되는지 도출한다.
import numpy as np

def numerical_derivative(f,x):
    delta_x=1e-4    #미세하게 움직이는 값
    grad=np.zeros_like(x)
    it=np.nditer(x,flags=['multi_index'],op_flags=['readwrite']) # input x값을 하나하나 불러온다.

    while not it.finished:
        idx=it.multi_index
        tmp_val=x[idx]

        x[idx]=float(tmp_val)+delta_x   #delta_x만큼 +로 움직였을 때
        fx1=f(x)

        x[idx]=tmp_val-delta_x    #delta_x만큼 -로 움직였을 때
        fx2=f(x)

        grad[idx]=(fx1-fx2)/(2*delta_x) #각 입력값에서 delta_x만큼 움직인 평균 거리 도출

        x[idx]=tmp_val

        it.iternext()

    return grad
