'''
Method to speed up training
1 Stochastic Gradient Descent (SGD): sperate data into small parts continously put into NN
2 Changing W+ = -learning rate*dx 
 2.1 Momentum m = b1*m - learning rate *dx, W + = m
 2.1 AdaGrad V + = dx^2, W+ = -learning rate *dx/root(V)
 2.3 RMSProp V = b1*V + (1-b1)*dx^2, W+ = -learning rate*dx/root(V)
 2.3 Adam m = b1*m + (1-b1)*dx, V = b2*V + (1-b2)*dx^2, W+ = -learning rate*m/root(V)
'''