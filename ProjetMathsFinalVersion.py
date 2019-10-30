# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 17:50:37 2019

@author: Perrotin
"""
import autograd
from autograd import numpy as np
import pylab as pl
import matplotlib 

'''
Fonction find seed qui se base sur un algo dichotomique
oublie pas de présiser les conditions raisonnables
'''
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

'''
Là on arrive sur la première version de simple contour, avec le théorème des fonctions implicites
'''
def grad_f(x,y):
    '''
    Calcule le gradient en un point
    '''
    g=autograd.grad
    return np.r_[g(f,0)(x,y),g(f,1)(x,y)]

def prochain_point(f,x,y,delta=0.01):
    '''
    on part de (x,y) et on détermine le point suivant avec le théorème
    '''
    grad=grad_f(x,y)
    if grad[1]!=0:
        d_phi=-grad[0]/grad[1]
        delta2=delta/sqrt(1+d_phi**2)
        return (x+delta2, y+d_phi*delta2)
    elif grad[0]!=0:
        d_phi=-grad[1]/grad[0]
        delta2=delta/sqrt(1+d_phi**2)
        return (x+d_phi*delta2,y+delta2)
    '''
    explique les disjonctions de cas ;-)
    '''
    else:
        return None
        
    
def simple_contour1(f, c=0, delta=0.01):
    def g(x):
        return f(0,x)
    t=find_seed(g,c)
    if t==None:
        return ([],[])
    else:
        les_x=[0]
        les_y=[t]
        
        while (1-les_x[-1])>delta or 1-les_y[-1]>delta: #il manque quelques conditions...
            point=prochain_point(f,les_x[-1],les_y[-1],delta)
            les_x.append(point[0])
            les_y.append(point[1])
        return (les_x,les_y)
    
'''
Problème de la méthode : on ne fait que avancer au niveau des x, il n'y a donc pas de rebroussement de 
chemin possible (imagine si on a une demie ellipse)
La méthode des gradients, comme il est orienté, permet de rebrousser chemin
'''

def newton(f,c,x,y,eps=2**-26):
    def g(x,y):
        return f(x,y)-c
    # g va nous permettre de se ramener à la recherche d'un point fixe
    def J_g(x,y):
        j=autograd.jacobian
        return np.c_[j(g,0)(x,y),j(g,1)(x,y)]
    J=J_g(x,y)
    #la jacobienne est ici un vecteur à deux composantes
    while g(x,y)>eps:
        #on arrête la boucle quand on est à esp du point fixe (même précision que find_seed)
        a=np.array((x,y))        
        a=a-J*g(x,y)
        x=a[0,0]
        y=a[0,1]
    return (x,y)
'''
la fonction qui suit est une version de simple_contour avec le gradient qui prend en compte l'orientation du gradient
mais cela ne semble pas utile pour la fonction contour_complexe qui apparament arrive à gérer ça toute seule
explique vite fait ça pour montrere qu'on a cerné pas mal de problème
'''
def simple_contour2.1(f,c=0.0,delta=0.01):
    def grad_f(x,y):
        g=autograd.grad
        return np.r_[g(f,0)(x,y),g(f,1)(x,y)]
    les_x=[0.0]
    g=lambda y:f(0,y)
    les_y=[find_seed(g,c)]
    if les_y[0]==None:
        return ([],[])
    elif grad_f(les_x[0],les_y[0])[1]>=0 :
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
'''
on enlève la disjonction de cas sur la gradient
'''
def simple_contour(f,c=0.0,delta=0.01):
    def grad_f(x,y):
        g=autograd.grad
        return np.r_[g(f,0)(x,y),g(f,1)(x,y)]
    les_x=[0.0]
    g=lambda y:f(0,y)
    les_y=[find_seed(g,c)]
    if les_y[0]==None:
        return ([],[])
    while ((1-les_x[-1])>delta and (1-les_y[-1])>delta and (les_x[-1])>delta and (les_y[-1])>delta) or (abs(les_x[0]-les_x[-1])<delta or abs(les_y[0]-les_y[-1])<delta)  :
        grad=grad_f(les_x[-1],les_y[-1])
        a=newton(f,c,les_x[-1]+grad[1]*delta,les_y[-1]-grad[0]*delta)
        les_x.append(a[0])
        les_y.append(a[1])
        '''
        On utilie la méthode de Newton pour redresser l'erreur dûe à l'approximation faite 
        par le calcul avec le gradient (on prend la droite tangente)
        '''
    #   les_x.append(les_x[-1]+grad[1]*delta)
    #   les_y.append(les_y[-1]-grad[0]*delta)
    return les_x,les_y
        
f1=lambda x,y:2*(np.exp(-x**2-y**2)-np.exp(-(x-1)**2-(y-1)**2))
f=lambda x,y:((x)**2+(y)**2)-.5**2+np.sin(x*y)*np.cos(2*x**2)
     
#a=simple_contour(f,c=0.)        
#X=a[0]
#Y=a[1]
#pl.plot(X,Y)
#pl.axis("equal")
#pl.show()   

'''
début du programme du prof
'''

LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3  # clockwise


def rotate_direction(direction, n=1):
    return (direction + n) % 4


def rotate(x, y, n=1):
    if n == 0:
        return x, y
    elif n >= 1:
        return rotate(1 - y, x, n - 1)
    else:
        assert n < 0
        return rotate(x, y, n=-3 * n)


def rotate_function(f, n=1):
    def rotated_function(x, y):
        xr, yr = rotate(x, y, -n)
        return f(xr, yr)

    return rotated_function


# Complex Contouring
# ------------------------------------------------------------------------------

# Customize the simple_contour function used in contour :
# simple_contour = smart_simple_contour


def contour(f, c, xs=[0.0, 1.0], ys=[0.0, 1.0], delta=0.01):
    curves = []
    nx, ny = len(xs), len(ys)
    for i in range(nx - 1):
        for j in range(ny - 1):
            xmin, xmax = xs[i], xs[i + 1]
            ymin, ymax = ys[j], ys[j + 1]

            def f_cell(x, y):
                return f(xmin + (xmax - xmin) * x, ymin + (ymax - ymin) * y)

            done = set()
            for n in [0, 1, 2, 3]:
                if n not in done:
                    rotated_f_cell = rotate_function(f_cell, n)
                    x_curve_r, y_curve_r = simple_contour(rotated_f_cell, c, delta)
                    exit = None
                    if len(x_curve_r) >= 1:
                        xf, yf = x_curve_r[-1], y_curve_r[-1]
                        if xf == 0.0:
                            exit = LEFT
                        elif xf == 1.0:
                            exit = RIGHT
                        elif yf == 0.0:
                            exit = DOWN
                        elif yf == 1.0:
                            exit = UP
                    if exit is not None:  # a fully successful contour fragment
                        exit = rotate_direction(exit, n)
                        done.add(exit)

                    x_curve, y_curve = [], []
                    for x_r, y_r in zip(x_curve_r, y_curve_r):
                        x, y = rotate(x_r, y_r, n=-n)
                        x_curve.append(x)
                        y_curve.append(y)
                    x_curve = np.array(x_curve)
                    y_curve = np.array(y_curve)
                    curves.append(
                        (xmin + (xmax - xmin) * x_curve, ymin + (ymax - ymin) * y_curve)
                    )
    return curves

'''
fin du programme du prof
'''

xs=np.linspace(-3,4,30)
ys=np.linspace(-3,4,30)
for i in range(5):
    level_curves = contour(f,i,xs,ys) 
    for x, y in level_curves:
        matplotlib.pyplot.plot(x, y, 'r')
        pl.axis("equal")