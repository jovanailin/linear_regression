import pandas as pd
import numpy as np
np.random.seed(1)
np.set_printoptions(formatter={'float_kind':lambda x: "%.3f" % x})

#%% PREPARE DATA
data = pd.read_csv('boston.csv')

y = data.iloc[:,[-1]] 
X = data.iloc[:,0:-1]


X_mean = X.mean()

X_std = X.std()

X = (X - X_mean) / X_std #normalizacija



X['X0'] = 1


y = y.as_matrix()
X = X.as_matrix()

n,m = X.shape


#print(n) # 506 broj slucajeva
#print(m) #14   broj parametara

#%% INITIALIZATION
w = np.random.random((1,m))
w_s=w[0]




alpha = 0.3 #learning rate, brzina konvergencije, sto je manji to sporije konvergira, ako je prevelik moze da divergira
beta = 1 #parameter regularizacije, odreduje jacinu regularizacije, previse mala vrednost
#regularizacija nece imati dovoljna efekta, previse velika vrednost, gradijentni spust nece uspeti da kovergira

##%% LEARN: GRADIENT DESCEND



for iter in range(1):
    h = X.dot(w.T) #h je vektor cost funkcija
    err=h-y #err je vektor mera odstupanja
    grad=err.T.dot(X)/n #grad je vektor parcijalnih izvoda funkcije greske
#    print("Grad")
#    print(grad)
#    print("w")
#    print(w)
    w = w*(1-alpha*beta/m) - alpha*grad
    mse=(err.T.dot(err))/n
    grad_norm = abs(grad).sum()
#    print("grad_norm",grad_norm)
#    print("mse", mse)
#    print("iter",iter)
#    print("alpha",alpha)
    if grad_norm<0.1 or mse<10: break


# STOCHASTIC GRADIENT DESCENT

   
np.random.shuffle(X)



for i in range(n):
        h=X[i].dot(w_s.T)
        print(h)
        err=h-y[i]
        s_grad = err*X[i]
        mse = err*err/n
        w_s = w_s - alpha*s_grad
        grad_norm = abs(s_grad).sum()
#        print("the last grad norm is")
#        print(grad_norm)
#        print("the last i")
#        print(i)
        if grad_norm<0.1 or mse<10: 
            break
       
        

#
#print(w)     
#print(w_s) 


##%% PREDICT
data_new = pd.read_csv('boston_novi.csv')
data_new = (data_new-X_mean)/X_std
data_new['X0'] = 1
prediction = data_new.as_matrix().dot(w.T)




