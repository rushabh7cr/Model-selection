import numpy as np
from numpy import genfromtxt
import time
start_time = time.time()

def model_selection(x,t):
    alpha = 6
    beta = 1
    x_transpose = np.transpose(x)
    
    a=0
    b=0
    while( abs(a-alpha)> 0.0000001 and abs(b-beta)>.000000001 ):
        
        a=alpha
        b=beta
        mat = beta*(x_transpose.dot(x))    
        eigenvalues = beta*np.linalg.eigvalsh(x_transpose.dot(x))                                 #calc of eignevalues for the symmetric matrix Φ^T.Φ  
        
        
        Sn =np.linalg.inv(alpha * np.identity(len(x_transpose)) + mat)                          # calc Sn 
        Mn = beta*(Sn.dot(x_transpose.dot(t)))                                                  #calc Mn        
        Mn_transpose = np.transpose(Mn)
        gamma=0    
        for i in range(len(eigenvalues)):                                                   #calculating gamma
             gamma+= eigenvalues[i] / (alpha + eigenvalues[i])   
            
        alpha = gamma / (Mn_transpose.dot(Mn))                                          #alpha
        
        t_hat=[]
        for i in range(len(x)):
            ti = 0
            for k in range(len(Mn)):
                ti += Mn[k]*x[i][k]                                                 
            t_hat.append(ti)
        e=0
        for i in range(len(t)):
            e += (t[i] - t_hat[i]) * (t[i] - t_hat[i])
            
        beta = (len(x)-gamma)/e
        
    return alpha/beta

def train(lamda,x,t):
    x_transpose = np.transpose(x)
    a = np.linalg.inv(x_transpose.dot(x) + (lamda * np.identity(len(x_transpose))))
    w = a.dot(x_transpose.dot(t))
    
    return w

def test(w,x,t):
    t_hat = []
    for i in range(len(x)):
        th = 0
        for k in range(len(w)):
            th += w[k] * x[i][k]
        t_hat.append(th)

    e = 0
    for i in range(len(t)):
        e += (t[i] - t_hat[i]) * (t[i] - t_hat[i])

    mse = (e / len(x))
    return mse   

if __name__ == "__main__":
    
    file = ('wine','crime','100-10','100-100','1000-100')
    for i in file:
        filename_train = 'train-' + i + '.csv'
        filename_trainR = 'trainR-' + i + '.csv'
        filename_test = 'test-' + i + '.csv'
        filename_testR = 'testR-' + i + '.csv'
    
        data = genfromtxt(filename_train,
                           delimiter=',',dtype=None)
        
        target = genfromtxt(filename_trainR,
                           delimiter=',',dtype=None)
        test_data = genfromtxt(filename_test,
                           delimiter=',',dtype=None)
        test_t = genfromtxt(filename_testR,
                           delimiter=',',dtype=None)
        lamda = model_selection(data,target)
        w = train(lamda,data,target)
        mse = test(w,test_data,test_t)
        
        print("The value of lambda for {} is: {}  mse : {}".format(i,lamda,mse))
    print("--- %s seconds ---" % (time.time() - start_time))    