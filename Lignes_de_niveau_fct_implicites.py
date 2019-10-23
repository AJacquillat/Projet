# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:10:07 2019

@author: Perrotin
"""
import autograd
from autograd import numpy as np
from math import sqrt
import pylab as pl

f=lambda x,y:(x**2+y**2)*1.0

def find_seed(f,a,b,x, c=0, eps=2**-26):
    milieu=(b+a)/2
    if (f(x,a)>c and f(x,b)>c) or (f(x,a)<c and f(x,b)<c):
        return None
    elif f(x,a)==c:
        return a
    elif f(x,b)==c:
        return b
    while (b-a)/2>eps:
        if f(d,milieu)>=c:
            b=milieu
        else:
            a=milieu
        milieu=(b+a)/2
    return milieu

#def g(x):
#    return x**2
#
#print(find_seed(g,-1,0,1))
    

def grad_f(x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def newton(f,c,x,y,eps=2**-26):
    def g(x,y):
        return f(x,y)-c
    def J_g(x,y):
        j=autograd.jacobian
        return np.c_[j(g,0)(x,y),j(g,1)(x,y)]
    J=J_g(x,y)
    while g(x,y)>eps:
        a=np.array((x,y))        
        a=a-J*g(x,y)
        x=a[0,0]
        y=a[0,1]
    return (x,y)

    
    

def prochain_point(f,c,x,y,delta=0.01,sens=1):
    grad=grad_f(x,y)
    if grad[1]!=0:
        d_phi=-grad[0]/grad[1]
        delta2=delta/sqrt(1+d_phi**2)
        return newton(f,c,x+delta2, y+d_phi*delta2)
#        return (x+delta2, y+d_phi*delta2)
    elif grad[0]!=0:
        d_phi=-grad[1]/grad[0]
        delta2=delta/sqrt(1+d_phi**2)
        return newton(f,c,x+d_phi*delta2,y+delta2)
#        return (x+d_phi*delta2,y+delta2)
    else:
        return None
        
    
def simple_contour(f, t,d,sens=1, c=1, delta=0.0001):
    if t==None:
        return ([],[])
    else:
        les_x=[d]
        les_y=[t]
        i=0
        while (abs(1-les_x[-1])>delta and abs(1-les_y[-1])>delta and abs(les_x[-1])>delta and abs(les_y[-1])>delta) or i<1/delta :
            i+=1
            point=prochain_point(f,c,les_x[-1],les_y[-1],delta,sens)
            if point==None:
                a=newton(f,c,les_x[-1]+delta, les_y[-1]+delta)
                les_x.append(a[0])
                les_y.append(a[1])                
            else :
               les_x.append(point[0])
               les_y.append(point[1])
        if sens==-1:
            les_y.reverse()
            les_x.reverse()
            for i in range(len(les_x)):
                les_x[i]=-les_x[i]
        return (les_x,les_y)
    
    
#print(find_seed(g,-1.0,0.,1))  

 
def contour(f,c=1):
    t1=find_seed(f,0.,1.,0.,c)
    t2=find_seed(f,-1.,0.,0.,c)
    t3=find_seed(f,0.,1.,-1.,c)
    t4=find_seed(f,-1.,0.,0.,c)
    print(t1,t2,t3,t4)
    a=simple_contour(f,t1,0.0)
    print(1)
    b=simple_contour(f,t2,0.0)
    print(2)
    c=simple_contour(f,t3,-1.)
    print(3)
    d=simple_contour(f,t4,0.,sens=-1)
    print(4)
    X1=a[0]
    Y1=a[1]
    X2=b[0]
    Y2=b[1]
    X3=c[0]
    Y3=c[1]
    X4=d[0]
    Y4=d[1]
    print(X4[0],Y4[0])
    pl.plot(X2,Y2)
    pl.plot(X1,Y1)
    pl.plot(X3,Y3)
    pl.plot(X4,Y4)
    pl.axis("equal")
    pl.show()
    return (X1+X2+X3+X4,Y1+Y2+Y3+Y4)

    
a=contour(f)        
X=a[0]
Y=a[1]
pl.plot(X,Y)
pl.axis("equal")
pl.show()
        