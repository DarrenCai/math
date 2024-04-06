# Copyright (c) 2024 DarrenCai
# All rights reserved.
#
# This code is licensed under the MIT License.

# -*- coding: utf-8 -*-

import numpy as np

eps, eps1 = 1e-12, 1e-10

def solve_quadratic_equation(a: float, b: float, c: float):
    '''
    Find the real roots of the quadratic equation.
    params: coefficients a, b, c (a can be 0)
    '''
    if a == 0:
        return [] if b == 0 else [-c/b]
    delta = b*b - 4*a*c
    if delta < 0:
        if delta < -eps1:
            return []
        delta = 0
    a *= 2; delta = delta**.5
    return [-b/a] if delta == 0 or delta < a*eps1 else [(-b-delta)/a, (delta-b)/a]

def cubic_root(x: float):
    '''
    Calculate x^1/3.
    Avoid RuntimeWarning: 'invalid value encountered in scalar power' 
    '''
    return -(-x)**(1/3) if x<0 else x**(1/3)

def solve_cubic_equation(a: float, b: float, c: float, d: float):
    '''
    Find the real roots of the univariable cubic equation.
    https://blog.csdn.net/u012912039/article/details/101363323
    params: coefficients a, b, c, d, e (a cannot be 0)
    '''
    A, B, C, k = b*b - 3*a*c, b*c - 9*a*d, c*c - 3*b*d, 1/3
    if A == 0 and B == 0:
        return [-c/b]
    delta = B*B - 4*A*C
    if delta > 0:
        delta = delta**.5; y1, y2 = A*b + 1.5*a*(delta-B), A*b - 1.5*a*(delta+B)
        return [-k * (cubic_root(y1) + cubic_root(y2) + b) / a]
    if delta > -eps:
        return [B/A - b/a, -B/2/A]
    t = np.arccos((A*b-1.5*a*B)/A**1.5)/3; s = 3**.5*np.sin(t); t = np.cos(t); A = A**.5
    return [-k * (b + 2*A*t) / a, k * (A*(t + s) - b) / a, k * (A*(t - s) - b) / a]

def find_a_root(a: float, b: float, c: float, d: float):
    '''
    Find a real root of the univariable cubic equation.
    params: coefficients a, b, c, d, e (a cannot be 0)
    '''
    print(f'a: {a}, b: {b}, c: {c}, d: {d}')
    A, B, C, k = b*b - 3*a*c, b*c - 9*a*d, c*c - 3*b*d, 1/3
    if A == 0 and B == 0:
        return -c/b
    delta = B*B - 4*A*C
    if delta > 0:
        delta = delta**.5; y1, y2 = A*b + 1.5*a*(delta-B), A*b - 1.5*a*(delta+B)
        return -k * (cubic_root(y1) + cubic_root(y2) + b) / a
    if delta > -eps:
        return -B/2/A
    return -k * (b + 2*A**.5*np.cos(np.arccos((A*b-1.5*a*B)/A**1.5)/3)) / a
