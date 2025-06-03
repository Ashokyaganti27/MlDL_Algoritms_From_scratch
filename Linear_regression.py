from logger import logger

import pandas  as pd
import numpy as np

def Linear_regression(x,y,ephocs):

    m_value=0
    c_value=0
    alpha_value=0.01
    old_total_error=float("inf")
    epsilon=1e-6
    for _ in range(ephocs):
        total_error=0
        m_gradient=0
        c_gradient=0
        

        for i in range(len(x)):

            y_hat=m_value*x[i]+c_value

            total_error+=(y[i]-y_hat)**2

            m_gradient+=x[i]*(y[i]-y_hat)

            c_gradient+=y[i]-y_hat

        if abs(total_error-old_total_error)<epsilon:

            return m_value,c_value

        logger.info(f"For {m_value:.3f} and {c_value:.3f} we got {total_error/len(x):.3f}")
        
        m_value=m_value-(alpha_value*(-2/len(x)*m_gradient))

        c_value=c_value-(alpha_value*(-2/len(x)*c_gradient))

        old_total_error=total_error


    return m_value,c_value
    
def data(path,ephocs):

    data=pd.read_csv(path)

    x=data.iloc[:,0]

    y=data.iloc[:,-1]

    data=Linear_regression(x,y,ephocs)

    return data

data("datset.csv",10)


