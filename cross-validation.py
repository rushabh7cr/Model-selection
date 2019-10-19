import numpy as np
from numpy import genfromtxt
import time
start_time = time.time()

def model_selection(data,target):
    
    if len(data)%10!=0:
        a=len(data)%10
        b=len(data) - a
        d=data[:b]
        e = data[b:]
        f = target[:b]
        g = target[b:]
        data_folds = np.vsplit(d,10)
        target_folds = np.split(f,10)
        data_folds[0] = np.vstack((data_folds[0],e))
        target_folds[0] = np.concatenate((target_folds[0],g))
            
            
    else:
        data_folds = np.vsplit(data,10)
        target_folds = np.split(target,10)
    
    
    MSE=[]
    for lamda in range(0,151):
        mse=0
        
        for i in range(len(data_folds)):
            t_hat = []
            test = data_folds[i]
            test_t=target_folds[i]
            
            x_train = data_folds[:i] + data_folds[i+1:]
            
            train = np.vstack((x_train))
            y_t = target_folds[:i] + target_folds[i+1:]
            train_t = np.concatenate(y_t)
            train_transpose = np.transpose(train)
            
            
                    # w = (λI + ΦTΦ)^−1 * ΦTt
            a = np.linalg.inv(train_transpose.dot(train) + (lamda * np.identity(len(train_transpose))))
            w = a.dot(train_transpose.dot(train_t))
            
            
            for i in range(len(test)):
                ti = 0
                for k in range(len(w)):
                    ti += w[k]*test[i][k]
                t_hat.append(ti)
            
                    # MSE for test set
            e=0
            for i in range(len(test_t)):
                e += (test_t[i] - t_hat[i]) * (test_t[i] - t_hat[i])
            
            mse += (e/len(test))
        MSE.append(mse/10)
    
    return (MSE.index(min(MSE)))


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
        
        print("Lambda for {} is: {} TEST MSE : {}".format(i,lamda,mse))
    print("--- %s seconds ---" % (time.time() - start_time))