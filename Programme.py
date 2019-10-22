# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:10:07 2019

@author: Perrotin
"""
import autograd
from autograd import numpy as np
from math import sqrt

def find_seed(g, c=0, eps=2**-26):
    a=0
    b=1
    milieu=(b+a)/2
    if (g(a)>c and g(b)>c) or (g(a)<c and g(b)<c):
        return None
    while (b-a)/2>eps:
        if g(milieu)>=c:
            b=milieu
        else:
            a=milieu
        milieu=(b+a)/2
    return milieu

#def g(x):
#    return (x-0.25)**5
#
#print(find_seed(g))
def f(x,y):
    return np.sin(x)+2*np.sin(y)

def grad_f(x,y):
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def prochain_point(f,x,y,delta=0.01):
    grad=grad_f(x,y)
    if grad[1]!=0:
        d_phi=-grad[0]/grad[1]
        delta2=delta/sqrt(1+d_phi**2)
        return (x+delta2, y+d_phi*delta2)
    elif grad[0]!=0:
        d_phi=-grad[1]/grad[0]
        delta2=delta/sqrt(1+d_phi**2)
        return (x+d_phi*delta2,y+delta2)
    else:
        return None
        
    
def simple_contour(f, c=0, delta=0.01):
    t=find_seed(f,c)
    if t==None:
        return ([],[])
    else:
        les_x=t[0]
        les_y=t[1]