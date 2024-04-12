# Copyright (c) 2024 DarrenCai
# All rights reserved.
#
# This code is licensed under the MIT License.

# -*- coding: utf-8 -*-

import numpy as np

eps, eps1, eps2 = 1e-5, 1e-10, 1e-12

class circle():
    ''' circle is defined by its center point (c) and radius (r) '''
    def __init__(self, c: tuple[float, float], r: float):
        self.c = c; self.r = r
    def point_on(self, x: float, y: float):
        ''' Check whether the point (x, y) is on the curve '''
        return abs(((x-self.c[0])**2 + (y-self.c[1])**2)**.5 / self.r - 1) < eps

class ellipse():
    ''' ellipse is defined by its two foci (f1, f2) and major axis lenth (ma) '''
    def __init__(self, f1: tuple[float, float], f2: tuple[float, float], ma: float):
        self.f1 = f1; self.f2 = f2; self.ma = ma
    def point_on(self, x: float, y: float):
        ''' Check whether the point (x, y) is on the curve '''
        x1, y1 = self.f1; x2, y2 = self.f2
        return abs((((x-x1)**2 + (y-y1)**2)**.5 + ((x-x2)**2 + (y-y2)**2)**.5) / self.ma - 1) < eps

class parabola():
    ''' parabola is defined by its focus (f) and directrix (l) '''
    def __init__(self, f: tuple[float, float], l: tuple[float, float, float]):
        self.f = f; self.l = l
    def point_on(self, x: float, y: float):
        ''' Check whether the point (x, y) is on the curve '''
        x1, y1 = self.f; a, b, c = self.l
        return abs(((x-x1)**2 + (y-y1)**2)**.5 - abs(a*x + b*y +c)/(a*a+b*b)**.5) < eps

class hyperbola():
    ''' hyperbola is defined by two foci (f1, f2) and real axis lenth (ra) '''
    def __init__(self, f1: tuple[float, float], f2: tuple[float, float], ra: float):
        self.f1 = f1; self.f2 = f2; self.ra = ra
    def point_on(self, x: float, y: float):
        ''' Check whether the point (x, y) is on the curve '''
        x1, y1 = self.f1; x2, y2 = self.f2
        return abs(abs(((x-x1)**2 + (y-y1)**2)**.5 - ((x-x2)**2 + (y-y2)**2)**.5) / self.ra - 1) < eps

def ellipse_homogeneous(a: float, b: float, tilt: float, x: float, y: float):
    '''
    Calculate the homogeneous form coefficients based on the major/minor semi-axis,
    tilt angle and center coordinates of the ellipse.
    a: major semi-axis
    b: minor semi-axis
    tilt: tilt angle in radian
    x: x-axis coordinate of center
    y: y-axis coordinate of center
    return: homogeneous form coefficients (A, B, C, D, E, F)
    '''
    sin, cos = np.sin(tilt), np.cos(tilt)
    # A = sin**2 + (b/a)**2*cos**2; B = 2*((b/a)**2 - 1)*sin*cos; C = cos**2 + (b/a)**2*sin**2
    # D = -(2*A*x + B*y); E = -(2*C*y + B*x); F = -(D*x + E*y)/2 - b*b
    A = (sin/b)**2 + (cos/a)**2; B = 2*(1/a**2 - 1/b**2)*sin*cos; C = (cos/b)**2 + (sin/a)**2
    D = -(2*A*x + B*y); E = -(2*C*y + B*x); F = -(D*x + E*y)/2 - 1
    return (A, B, C, D, E, F)

def ellipse_params(h: tuple[float, float, float, float, float, float]):
    '''
    Calculate the major/minor semi-axis, tilt angle and center coordinates
    based on the homogeneous form coefficients of the ellipse.
    h: homogeneous form coefficients (A, B, C, D, E, F)
    return: major/minor semi-axis, tilt angle and center coordinates (a, b, tilt, cx, cy)
    '''
    A, B, C, D, E, F = h
    if A < 0:
        A = -A; B = -B; C = -C; D = -D; E = -E; F = -F
    p, q = 4*A*C - B*B, ((A-C)**2 + B*B)**.5; r = 2*((A*E*E - B*D*E + C*D*D)/p - F)
    return ((r / (A+C-q))**.5, (r / (A+C+q))**.5, np.arctan2(C - A - q, B), (B*E - 2*C*D) / p, (B*D - 2*A*E) / p)

def homogeneous_to_mat(h: tuple[float, float, float, float, float, float]):
    '''
    Generate 3x3 matrix representation of the conic based on its homogeneous form coefficients.
    h: homogeneous form coefficients (A, B, C, D, E, F)
    '''
    A, B, C, D, E, F = h
    return np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])

def parabola_homogeneous(x: float, y: float, f: float, tilt: float):
    '''
    Calculate the homogeneous form coefficients based on vertex (x, y),
    focal length and tilt angle of the parabola.
    x: x coordinate of vertex
    y: y coordinate of vertex
    f: focal length
    tilt: tilt angle in radian
    return: homogeneous form coefficients (A, B, C, D, E, F)
    '''
    sin, cos = np.sin(tilt), np.cos(tilt)
    A, B, C = sin**2, -2*sin*cos, cos**2; D, E = -2*A*x-B*y-4*f*cos, -2*C*y-B*x-4*f*sin
    return (A, B, C, D, E, A*x*x + B*x*y + C*y*y + 4*f*x*cos + 4*f*y*sin)

def hyperbola_homogeneous(a: float, b: float, tilt: float, x: float, y: float):
    '''
    Calculate the homogeneous form coefficients based on the real/imaginary semi-axis,
    tilt angle and center coordinates of the hyperbola.
    a: real semi-axis
    b: imaginary semi-axis
    tilt: tilt angle in radian
    x: x-axis coordinate of center
    y: y-axis coordinate of center
    return: homogeneous form coefficients (A, B, C, D, E, F)
    '''
    sin, cos = np.sin(tilt), np.cos(tilt)
    A = (cos/a)**2 - (sin/b)**2; B = 2*(1/a**2 + 1/b**2)*sin*cos; C = (sin/a)**2 - (cos/b)**2
    D = -(2*A*x + B*y); E = -(2*C*y + B*x); F = -(D*x + E*y)/2 - 1
    return (A, B, C, D, E, F)

def circle_from_homogeneous(A: float, B: float, C: float, D: float, E: float, F: float):
    '''
    Construct circle based on its homogeneous form coefficients.
    A = C and B = 0 and A!=0
    '''
    return circle((-D/2/A, -E/2/A), ((D*D + E*E)/4/A/A - F/A)**.5)

def ellipse_from_homogeneous(A: float, B: float, C: float, D: float, E: float, F: float):
    '''
    Construct ellipse based on its homogeneous form coefficients.
    B*B - 4*A*C > 0
    '''
    if A < 0:
        A = -A; B = -B; C = -C; D = -D; E = -E; F = -F
    p, q = 4*A*C - B*B, ((A-C)**2 + B*B)**.5; r = 2*((A*E*E - B*D*E + C*D*D)/p - F)
    a, b, tilt = (r / (A+C-q))**.5, (r / (A+C+q))**.5, np.arctan2(C-A-q, B)
    c, x, y = (a*a - b*b)**.5, (B*E - 2*C*D) / p, (B*D - 2*A*E) / p; dx, dy = c*np.cos(tilt), c*np.sin(tilt)
    return ellipse((x-dx, y-dy), (x+dx, y+dy), 2*a)

def parabola_from_homogeneous(A: float, B: float, C: float, D: float, E: float, F: float):
    '''
    Construct parabola based on its homogeneous form coefficients.
    B*B - 4*A*C = 0
    '''
    tilt = np.arctan2(2*A, -B); cos, sin = np.cos(tilt), np.sin(tilt); s, t = (A+C)*cos, (A+C)*sin
    f = (B*E-2*C*D) / (8*C*s-4*B*t)
    if f < 0:
        tilt = np.arctan2(-2*A, B); f, sin, cos, s, t = -f, -sin, -cos, -s, -t
    D1, E1 = D + 4*f*s, E + 4*f*t; s, t = 4*f*s - D1/2, 4*f*t - E1/2
    x, dx, dy = (B*F + D1*t) / (B*s - 2*A*t), f*cos, f*sin; y = -(2*A*x + D1)/B; c = (dx-x)*cos + (dy-y)*sin
    print(f'vertex: ({x}, {y}), focal: {f}, tilt: {np.rad2deg(tilt)}')
    return parabola((x + dx, y + dy), (cos, sin, c))

def hyperbola_from_homogeneous(A: float, B: float, C: float, D: float, E: float, F: float):
    '''
    Construct hyperbola based on its homogeneous form coefficients.
    B*B - 4*A*C < 0
    '''
    p, q = 4*A*C - B*B, ((A-C)**2 + B*B)**.5; r = 2*((A*E*E - B*D*E + C*D*D)/p - F)
    a, b, tilt = (r / (q+A+C))**.5, (r / (q-A-C))**.5, np.arctan2(C-A+q, B)
    c, x, y = (a*a + b*b)**.5, (B*E - 2*C*D) / p, (B*D - 2*A*E) / p; dx, dy = c*np.cos(tilt), c*np.sin(tilt)
    return hyperbola((x-dx, y-dy), (x+dx, y+dy), 2*a)

def geo_from_mat(q):
    '''
    Calculate the conic's geometric definition based on its matrix representation.
    q: the input 3x3 matrix
    '''
    A, B, C, D, E, F = q[0,0], 2*q[0,1], q[1,1], 2*q[0,2], 2*q[1,2], q[2,2]
    if A == C and B == 0:
        return circle_from_homogeneous(A, B, C, D, E, F)
    a33 = np.linalg.det(q[:2,:2])
    if a33 > 0:
        return ellipse_from_homogeneous(A, B, C, D, E, F)
    if a33 == 0:
        return parabola_from_homogeneous(A, B, C, D, E, F)
    return hyperbola_from_homogeneous(A, B, C, D, E, F)

def ellipse_points(a: float, b: float, tilt: int, x: float, y: float):
    '''
    Generate the points based on the major/minor semi-axis, tilt angle and center coordinates of the ellipse.
    a: major semi-axis
    b: minor semi-axis
    tilt: tilt angle in degree
    x: x-axis coordinate of center
    y: y-axis coordinate of center
    '''
    px, py, tilt = [], [], np.deg2rad(tilt); cos, sin = np.cos(tilt), np.sin(tilt)
    rot = np.array([[cos, -sin], [sin, cos]])
    for t in range(361):
        point = np.array([a * np.cos(np.deg2rad(t)), b * np.sin(np.deg2rad(t))])
        rot_p = np.dot(rot, point.T); px.append(rot_p[0] + x); py.append(rot_p[1] + y)
    return px, py

def parabola_points(x: float, y: float, f: float, tilt: int):
    '''
    Generate the points based on the vertex (x, y), focal lenth and tilt angle of the parabola.
    x: x coordinate of vertex
    y: y coordinate of vertex
    f: focal length
    tilt: tilt angle in degree
    '''
    ox, oy, px, py = [.05*x for x in range(200, -1, -1)], [], [], []
    for i in range(201):
        oy.append((ox[i]/4/f)**.5)
    for i in range(199, -1, -1):
        ox.append(ox[i]); oy.append(-(ox[i]/4/f)**.5)
    t = np.deg2rad(tilt); cos, sin = np.cos(t), np.sin(t)
    for i in range(401):
        px.append(x + ox[i]*cos - oy[i]*sin); py.append(y + oy[i]*cos + ox[i]*sin)
    return px, py

def hyperbola_points(a: float, b: float, tilt: int, x: float, y: float):
    '''
    Generate the points based on the real/imaginary semi-axis, tilt angle and center coordinates of the hyperbola.
    a: real semi-axis
    b: imaginary semi-axis
    tilt: tilt angle in degree
    x: x-axis coordinate of center
    y: y-axis coordinate of center
    '''
    t, tilt = (2*max(a, b) - a) / 200, np.deg2rad(tilt); cos, sin = np.cos(tilt), np.sin(tilt)
    ax, ay, bx, by = [a+(200-i)*t for i in range(201)], [], [], []
    for i in range(201):
        ay.append(b*((ax[i]/a)**2 - 1)**.5)
    for i in range(200):
        ax.append(ax[199-i]); ay.append(-ay[199-i])
    for i in range(401):
        tx = ax[i]; ty = ay[i]; ax[i] = x + tx*cos - ty*sin; ay[i] = y + ty*cos + tx*sin
    for i in range(401):
        bx.append(2*x - ax[i]); by.append(2*y - ay[i])
    return (ax, ay, bx, by)

def cofactor(p):
    '''
    Calculate the cofactor matrix of p.
    p: the input matrix
    '''
    rows, cols = p.shape; c = np.zeros_like(p)
    for i in range(rows):
        for j in range(cols):
            c[i,j] = (-1 if (i^j)&1 else 1) * np.linalg.det(p[np.arange(rows)!=i][:,np.arange(cols)!=j])
    return c

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
    print(f'a: {a}, b: {b}, c: {c}, d: {d}')
    A, B, C, k = b*b - 3*a*c, b*c - 9*a*d, c*c - 3*b*d, 1/3
    if A == 0 and B == 0:
        return [-c/b]
    delta = B*B - 4*A*C
    if delta > 0:
        delta = delta**.5; y1, y2 = A*b + 1.5*a*(delta-B), A*b - 1.5*a*(delta+B)
        return [-k * (cubic_root(y1) + cubic_root(y2) + b) / a]
    if delta > -eps2:
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
    if delta > -eps2:
        return -B/2/A
    return -k * (b + 2*A**.5*np.cos(np.arccos((A*b-1.5*a*B)/A**1.5)/3)) / a

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

def line_conic_intersection(q, a: float, b: float, c: float):
    '''
    Find the intersection points with line ax+by+c=0 and conic section according to its matrix representation.
    q: 3x3 matrix representing the conic
    '''
    if a == 0 and b == 0:
        return []
    if abs(a) > abs(b):
        d, e = q[0,0]*b*b - 2*q[0,1]*a*b + q[1,1]*a*a, 2*(q[1,2]*a*a + q[0,0]*b*c - q[0,1]*a*c - q[0,2]*a*b)
        r = solve_quadratic_equation(d, e, q[0,0]*c*c - 2*q[0,2]*a*c + q[2,2]*a*a)
        return [((-c-b*y)/a, y) for y in r]
    d, e = q[0,0]*b*b - 2*q[0,1]*a*b + q[1,1]*a*a, 2*(q[0,2]*b*b + q[1,1]*a*c - q[0,1]*b*c - q[1,2]*a*b)
    r = solve_quadratic_equation(d, e, q[1,1]*c*c - 2*q[1,2]*b*c + q[2,2]*b*b)
    return [(x, (-c-a*x)/b) for x in r]

def check(g: circle | ellipse | parabola | hyperbola, r: list[tuple[float, float]]):
    '''
    Remove duplicate points and points that are outside conic section according to its matrix representation.
    g: geometry of the conic: circle, ellipse, parabola, hyperbola
    r: candidate points
    '''
    p = []
    for x, y in r:
        if g.point_on(x, y) and np.sum([abs(p[i][0]-x) < eps1 and abs(p[i][1]-y) < eps1 for i in range(len(p))]) == 0:
            p.append((x, y))
    return p

def conics_intersection(p, q):
    '''
    Find the intersection points of two conics according to their matrix representation.
    https://en.wikipedia.org/wiki/Matrix_representation_of_conic_sections
    https://en.wikipedia.org/wiki/Conic_section#Intersecting_two_conics
    p: 3x3 matrix representing the first conic
    q: 3x3 matrix representing the second conic
    '''
    r = find_a_root(np.linalg.det(p), np.trace(np.dot(cofactor(p), q)), np.trace(np.dot(p, cofactor(q))), np.linalg.det(q))
    s = r*p + q; a, b, c, d, e, f, a33 = s[0,0], s[0,1], s[1,1], s[0,2], s[1,2], s[2,2], np.linalg.det(s[:2,:2])
    gp, gq = geo_from_mat(p), geo_from_mat(q)
    if a33 > eps2:
        x, y = (b*e - c*d) / a33, (b*d - a*e) / a33
        return [(x, y)] if gp.point_on(x, y) and gq.point_on(x, y)  else []
    if a33 < -eps2:
        a33 = (-a33)**.5; c = (b*d - a*e) / a33
        r1 = line_conic_intersection(q, a, b-a33, d-c)
        for r in line_conic_intersection(q, a, b+a33, d+c):
            r1.append(r)
        return check(gp, r1)
    if abs(b) < eps2:
        if abs(a) > max(abs(c), eps2):
            x, r1 = solve_quadratic_equation(a, 2*d, f), []
            if len(x) > 0:
                r1 = line_conic_intersection(q, -1, 0, x[0])
            if len(x) > 1:
                for r in line_conic_intersection(q, -1, 0, x[1]):
                    r1.append(r)
            return check(gp, r1)
        if abs(c) > max(abs(a), eps2):
            y, r1 = solve_quadratic_equation(c, 2*e, f), []
            if len(y) > 0:
                r1 = line_conic_intersection(q, 0, -1, y[0])
            if len(y) > 1:
                for r in line_conic_intersection(q, 0, -1, y[1]):
                    r1.append(r)
            return check(gp, r1)
        return check(gp, line_conic_intersection(q, 2*d, 2*e, f))
    if a < 0:
        a = -a; b = -b; c = -c; d = -d; e = -e; f = -f
    s = d*d - a*f
    if b*d*e < -eps2 or abs(a*e*e - c*d*d) > eps2 or s < -eps2:
        return []
    if s < eps:
        return check(gp, line_conic_intersection(q, a, b, d))
    s = s**.5; r1 = line_conic_intersection(q, a, b, d-s)
    for r in line_conic_intersection(q, a, b, d+s):
        r1.append(r)
    return check(gp, r1)

def test_parabola():
    ''' test conveting between homogeneous form coefficients and parabola params ( vertex, focal length and tilt angle ) '''
    x, y, f, tilt = random()*6-3, random()*6-3, random()*6, int(random()*361); t = np.deg2rad(tilt)
    print(f'vertex: ({x}, {y}) focal: {f}, tilt: {tilt}')
    print(f'fx: {x+f*np.cos(t)}, fy: {y+f*np.sin(t)}, a: {np.cos(t)}, b: {np.sin(t)}, '
          f'c: {(f*np.cos(t)-x)*np.cos(t) + (f*np.sin(t)-y)*np.sin(t)}')
    px, py = parabola_points(x, y, f, tilt)
    plt.clf(); plt.plot(px, py, 'b-', zorder=1, linewidth=1)
    A, B, C, D, E, F = parabola_homogeneous(x, y, f, t)
    print(f'homogeneous: {A}, {B}, {C}, {D}, {E}, {F}')
    parabola = parabola_from_homogeneous(A, B, C, D, E, F)
    fx, fy = parabola.f; a, b, c = parabola.l
    print(f'fx: {fx}, fy: {fy}, a: {a}, b: {b}, c: {c}\n')
    plt.scatter(fx, fy, marker='o', c='red', s=20, zorder=2)
    if abs(b) > abs(a):
        sx = [0.1*x for x in range(-100, 101, 1)]; sy = [-(a*x+c)/b for x in sx]
    else:
        sy = [0.1*y for y in range(-100, 101, 1)]; sx = [-(b*y+c)/a for y in sy]
    plt.plot(sx, sy, 'r-', zorder=1, linewidth=1); plt.show()

def test_hyperbola():
    ''' test conveting between homogeneous form coefficients and hyperbola params ( real/imaginary semi-axis, tilt angle and center ) '''
    a, b, tilt, x, y = max(random()*8, .5), max(random()*8, .5), int(random()*361), random()*24 - 12, random()*24 - 12
    A, B, C, D, E, F = hyperbola_homogeneous(a, b, np.deg2rad(tilt), x, y)
    print(f'a: {a}, b: {b}, tilt: {tilt}, x: {x}, y: {y}')
    print(f'homogeneous: {A}, {B}, {C}, {D}, {E}, {F}')
    ax, ay, bx, by = hyperbola_points(a, b, tilt, x, y)
    plt.clf(); plt.plot(ax, ay, 'b-', zorder=1, linewidth=1); plt.plot(bx, by, 'b-', zorder=1, linewidth=1)
    hyperbola = hyperbola_from_homogeneous(A, B, C, D, E, F)
    a1 = hyperbola.ra/2; x1, y1 = hyperbola.f1; x2, y2 = hyperbola.f2; b1 = (((x2-x1)**2 + (y2-y1)**2) / 4 - a1**2)**.5
    print(f'a: {a1}, b: {b1}, tilt: {np.rad2deg(np.arctan2(y2-y1, x2-x1))}, x: {(x1+x2)/2}, y: {(y1+y2)/2}\n')
    plt.show()

def test_intersection(e1, e2, f = None):
    ''' test conics intersection '''
    
    def convert(dat):
        a, b, tilt, x, y = dat
        h = ellipse_homogeneous(a, b, np.deg2rad(tilt), x, y)
        print(f'a: {a}, b: {b}, tilt: {tilt}, x: {x}, y: {y}')
        print(f'homogeneous: {h}')
        a1, b1, tilt1, x1, y1 = ellipse_params(h)
        print(f'a: {a1}, b: {b1}, tilt: {np.rad2deg(tilt1)}, x: {x1}, y: {y1}')

        px, py = ellipse_points(a, b, tilt, x, y)
        plt.plot(px, py, 'b-', zorder=1, linewidth=1)

        return homogeneous_to_mat(h)

    plt.clf()
    p, q = convert(e1), convert(e2)
    for x, y in conics_intersection(p, q):
        print(f'{x} {y}')
        plt.scatter(x, y, marker='o', c='red', s=20, zorder=2)
    print()
    f and plt.savefig(f)
    not f and plt.show()

if __name__ == '__main__':
    import os
    from random import random
    from matplotlib import pyplot as plt
    plt.rcParams["figure.figsize"] = [10, 10]
    plt.xlim(-20, 20)
    plt.ylim(-20, 20)

    os.makedirs('computational_geometry/conic_sections/test_results', exist_ok=True)

    # for i in range(50):
    #     test_parabola()

    # for i in range(50):
    #     test_hyperbola()

    # f = open('computational_geometry/conic_sections/test_data.txt', 'w')
    # data = [(5, 1, 270, -0.0244, 2), (5, 4, 270, 3, 0)]
    # for i in range(400):
    #     a = int(random()*5) + 1; b = int(random()*a) + 1; tilt = 90*int(random()*5)
    #     x = int(random()*7) - 3; y = int(random()*7) - 3; data.append((a, b, tilt, x, y))
    #     f.write(f'{a} {b} {tilt} {x} {y}\n')
    # for i in range(2000):
    #     a = max(random()*8, .5); b = max(random()*a, .5); tilt = int(random()*361)
    #     x = random()*24 - 12; y = random()*24 - 12; data.append((a, b, tilt, x, y))
    #     f.write(f'{a} {b} {tilt} {x} {y}\n')

    data = [
            [float(v) for v in line.split(' ')] for line in 
                open('computational_geometry/conic_sections/test_data.txt', 'r').read().splitlines()
        ]
    
    for i in range(0, len(data), 2):
        test_intersection(data[i], data[i+1], f'computational_geometry/conic_sections/test_results/{i//2+1}.png')