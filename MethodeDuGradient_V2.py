# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:50:37 2019

@author: Perrotin
"""
import autograd
from autograd import numpy as np
import pylab as pl


def find_seed(g, c=0, eps=2**-26):
    a=0
    b=1
    milieu=(b+a)/2
    if (g(a)>c and g(b)>c) or (g(a)<c and g(b)<c):
        return None
    while (b-a)/2>eps:
        if (g(milieu)-c)*(g(a)-c)<=0:
            b=milieu
        else:
            a=milieu
        milieu=(b+a)/2
    return milieu

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

def simple_contour(f,c=0.0,delta=0.01):
    def grad_f(x,y):
        g=autograd.grad
        return np.r_[g(f,0)(x,y),g(f,1)(x,y)]
    les_x=[0.0]
    g=lambda y:f(0,y)
    les_y=[find_seed(g,c)]
    if grad_f(les_x[0],les_y[0])[1]>=0 :
        while ((1-les_x[-1])>delta and (1-les_y[-1])>delta and (les_x[-1])>delta and (les_y[-1])>delta) or (abs(les_x[0]-les_x[-1])<delta or abs(les_y[0]-les_y[-1])<delta)  :
            grad=grad_f(les_x[-1],les_y[-1])
            a=newton(f,c,les_x[-1]+grad[1]*delta,les_y[-1]-grad[0]*delta)
            les_x.append(a[0])
            les_y.append(a[1])
            '''
            On utilie la méthode de Newton pour redresser l'erreur dûe à l'approximation faite 
            par le calcul avec le gradient (on prend la droite tangente)
            '''
    #        les_x.append(les_x[-1]+grad[1]*delta)
    #        les_y.append(les_y[-1]-grad[0]*delta)
        return les_x,les_y
    else :
        f1=lambda x,y:-f(x,y)
        def grad_f1(x,y):
            g=autograd.grad
            return np.r_[g(f1,0)(x,y),g(f1,1)(x,y)]
        les_x=[0.0]
        g=lambda y:f1(0,y)
        les_y=[find_seed(g,-c)]
        while ((1-les_x[-1])>delta and (1-les_y[-1])>delta and (les_x[-1])>delta and (les_y[-1])>delta) or (abs(les_x[0]-les_x[-1])<delta or abs(les_y[0]-les_y[-1])<delta)  :
            grad=grad_f1(les_x[-1],les_y[-1])
            a=newton(f1,-c,les_x[-1]+grad[1]*delta,les_y[-1]-grad[0]*delta)
            les_x.append(a[0])
            les_y.append(a[1])
        return les_x,les_y
        
f1=lambda x,y:2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))
f=lambda x,y:(x**2+y**2)-.5**2
     
a=simple_contour(f1,c=1.5)        
X=a[0]
Y=a[1]
pl.plot(X,Y)
pl.axis("equal")
pl.show()   