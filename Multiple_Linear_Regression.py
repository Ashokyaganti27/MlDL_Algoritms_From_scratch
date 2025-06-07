import pandas as pd
import numpy as np
from logger import logger
def multiple_linear_regression(x,y,epsilon=1e-6,epochs=5):

    features=x.shape[1]
    w=[]
    for i in range(features): # [ 0 for _ in range(features) ]
        w.append(0)
    n=x.shape[0]
    old_error=float("inf")
    b=0
    alpha=0.001
    for epoch in range(epochs):
        w_grad=[ 0 for _ in range(len(w))] # w_grad=[0]*features
        b_grad=0
        total_error=0
        for i in range(n):
            y_hat=0
    #          loss for single row
            for j in range(len(w)):
                y_hat+=w[j]*x[i][j] #we can also use sum([ w[j]*x[i][j]  for j in range(len(w))]) + b
            y_hat+=b                #np.dot(x[i],w)+b
            total_error+=(y[i]-y_hat)**2 # error=y-y_hat then we use these in y-y_hat everyplace 
    #         for gradients summation 
            for j in range(len(w)):
                w_grad[j]+=x[i][j]*(y[i]-y_hat)
            b_grad+=y[i]-y_hat

        if abs(total_error-old_error)<epsilon:
            return w,b

        logger.info(f" for epoch {epoch} total_error {abs(total_error/n)},m_value {w} b {abs(b)}")

            #     for updating weights after 1 epoch
        for j in range(len(w)):
            w[j]=w[j]-(alpha*(-2/n*w_grad[j]))
        b=b-(alpha*(-2/n*b_grad))
        old_error=total_error

    logger.info(f"All epochs completed {epochs}")
    return w,b




def data(path):
    data=pd.read_csv(path)

    x=data.iloc[:,:-1].values
    y=data.iloc[:,-1].values

    data=multiple_linear_regression(x,y,epochs=10)

    return data
data("D:\classification model selection\Machine Learning A-Z (Model Selection)\Regression\Data.csv")

