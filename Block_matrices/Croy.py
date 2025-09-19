import numpy as np
from numpy.linalg import svd
from numba import njit, prange
from numba.typed import List
from scipy.signal import hilbert
import os
import sys
file_p = (__file__[:-8])
sys.path.append(file_p)
from Croyk0nfig import CACHE, PARALLEL, FASTMATH, dE_fitting, L_cutoff, Gamma_fallback, Gamma_default
from Croyk0nfig import direct_eig_n, direct_eig_nW, direct_eig_lbtol, direct_eig_maxit, direct_eig_fact, direct_eig_eps
#,n=2, nW=2, eps=0.001, fact = 0.75, maxit=500, lbtol = 0.0

# In this python script many functions for fitting Lorentzians are defined. You will
# find many overlap integrals between Lorentzian functions and hat functions, and their
# Derivatives. These are then combined at some point to give the error of deviation between
# the Lorentzian fit and the function f defined in the 'hat' basis (ie the linear interpolant
# of f from some samplings f(xi) of it).
# This script also contains the gradient of this error function, which is important for speed in 
# the minimisation algorithm which inevitably will be done in order to make a good fit.
# The errors are implemented in terms of what actually changes when the Lorentzians are moved,
# meaning the term 
#             Int f(x)^2 dx
# is intentionally left out
#
# Many places you will find the numba njit decorator, which you can search the internet to find out what is
# but in essence it just makes the code, which is written in large in terms of for loops, fast.
# 
# 15.12.22 added tolerance keyword to three evaluation functions
# 14-17/4 added direct eigenvalue computation to enforce the requirement of the Gammas to be positive
# semi-definite.

atan = np.arctan
ln = np.log

# how far the linear ramp should go beyond the sampling interval
# its hardcoded here

offset_E = 0.0 + dE_fitting
Lrz_cutoff  = 0.0 + L_cutoff

DE_n, DE_nW, DE_lbtol, DE_maxit, DE_fact, DE_eps \
    =\
    0+direct_eig_n, 0+direct_eig_nW, 0.0+direct_eig_lbtol, 0+direct_eig_maxit, 0.0 + direct_eig_fact, 0.0 + direct_eig_eps

#print('Offset_E: ', offset_E)
# The basic basis function we might want to project onto
# Its a Lorentzian and takes the apex value of 1 at E = ei
# Its linewidth is  defined by W
@njit(cache = CACHE, fastmath = FASTMATH)
def L(E, W, ei):
    return W**2 /( (E - ei)**2  + W**2 )

# vectorised version of the above
@njit(cache = CACHE, fastmath = FASTMATH)
def Lvec(E, W, ei):
    return W**2 /( (E - ei)**2  + W**2 )

# Overlap between two Lorentzians
# Used in the Punish overlap rutine
@njit(cache = CACHE, fastmath = FASTMATH)
def Overlap(ei, wi, ej, wj):
    #if wi<0 or wj<0:
        #print('Intermediate Gamma_i less than zero, CHECK END RESULT!!!!')
    xi = ei + 1j*abs(wi)
    xj = ej + 1j*abs(wj)
    if abs(xi -  xj)< 1e-12:
        res = np.pi * abs(wi)/2
    else:
        res = 2j * np.pi/4 * wi * wj *( 1/(xj-np.conj(xi)) + 1/(xi - np.conj(xj)) - 1/(xi - xj) - 1/(xj - xi))
    return res.real
    
# Finite difference implementation of the various derivatives
@njit(cache = CACHE, fastmath = FASTMATH)
def d_Overlap_d_eps(ei, wi, ej, wj):
    eps = 1e-5
    return (Overlap(ei + eps, wi, ej, wj) - Overlap(ei - eps, wi, ej, wj)) /(2 * eps)
@njit(cache = CACHE, fastmath = FASTMATH)
def d_Overlap_d_gamma(ei, wi, ej, wj):
    eps = 1e-5
    return (Overlap(ei, wi + eps, ej, wj) - Overlap(ei, wi - eps, ej, wj)) /(2 * eps)

@njit(cache = CACHE, fastmath = FASTMATH)
def d_Overlap_d_eps_same(ei,wi):
    eps = 1e-5
    return (Overlap(ei + eps, wi, ei + eps, wi) - Overlap(ei - eps, wi, ei - eps, wi)) /(2 * eps)
@njit(cache = CACHE, fastmath = FASTMATH)
def d_Overlap_d_gamma_same(ei,wi):
    eps = 1e-5
    return (Overlap(ei, wi + eps, ei, wi + eps) - Overlap(ei, wi - eps, ei, wi - eps)) /(2 * eps)


# Overlap integral of a hat function defined by the three points x1,x2,x3 and a Lorentzian centered at 
# ej and with width gj

@njit(cache = CACHE, fastmath = FASTMATH)
def hat_with_lrtz(x1,x2,x3, ej, gj):
    #print(x2-x1, x3-x2)
    t1 = (ej - x1)/(x2-x1)  * (-gj) * (atan((ej-x2)/gj) - atan((ej-x1)/gj))
    t2 = gj**2/(2 *(x2-x1)) * (ln((x2 - ej)**2 + gj**2) - ln((x1 - ej)**2 + gj**2))
    t3 = (ej - x3)/(x3-x2)  * (-gj) * (atan((ej-x3)/gj) - atan((ej-x2)/gj))
    t4 = gj**2/(2 *(x3-x2)) * (ln((x3 - ej)**2 + gj**2) - ln((x2 - ej)**2 + gj**2))
    res = t1+t2-t3-t4
    return res.real

# Numerical derivative of the above
@njit(cache = CACHE, fastmath = FASTMATH)
def d_hat_with_lrtz_d_eps(x1,x2,x3,ej,gj):
    eps = 1e-5
    return (hat_with_lrtz(x1,x2,x3, ej + eps, gj) - hat_with_lrtz(x1,x2,x3, ej - eps, gj)) /(2 * eps)

@njit(cache = CACHE, fastmath = FASTMATH)
def d_hat_with_lrtz_d_gamma(x1,x2,x3,ej,gj):
    eps = 1e-5
    return (hat_with_lrtz(x1,x2,x3, ej, gj + eps) - hat_with_lrtz(x1,x2,x3, ej, gj - eps)) /(2 * eps)
@njit(cache = CACHE, fastmath = FASTMATH)
def Overlap_Hilbert_Lrz(Ei,Gi,Ej,Gj):
    # Find the missing factor of an imaginary unit
    ei, gi, ej, gj = 0.0, 0.0, 0.0, 0.0
    ei += Ei
    gi += Gi
    ej += Ej
    gj += Gj
    
    pi = ei + 1j * gi
    pj = ej + 1j * gj
    #print(abs(pi - pj))

    
    f11 = 1j * gi
    f12 = pi - ej
    f21 = 1j * gj
    f22 = pj - ei
    
    F11 = -1j/(pi - ej - 1j * gj)
    F12 = +1j/(pi - ej + 1j * gj)
    F21 = -1j/(pj - ei - 1j * gi)
    F22 = +1j/(pj - ei + 1j * gi)
    
    res1 = f11 * f12 * (F11 + F12)
    res2 = f21 * f22 * (F21 + F22)
    # missing an factor 1j to be consistent with cauchy formula
    return np.real(2 * np.pi * (res1 + res2)/4)

# EI, GI = 2.0, 1.0
# EJ, GJ = 2.0, 1.0
# def LI(x):
#     return (x - EI) * GI/((x-EI)**2 + GI**2)
# def LJ(x):
#     return (x - EJ) * GJ/((x-EJ)**2 + GJ**2)
# def test(x):
#     return LI(x)*LJ(x)
# from scipy.integrate import quad
# res1 = quad(test, -np.inf, np.inf)
# res2 = Overlap_Hilbert_Lrz(EI + 1e-10, GI+ 10e-10, EJ, GJ)



# An error function for the hilbert transform
# xi are the sampled points, 
# fi are the sampled values of x
# PARS is a (3,N) array containing the collection of linewidths (WI),
# the centres (EI) and the weights (GI).
# It returns the error of the 
def delta_error_linear_hilbertt_lrz(xi, fi, PARS):
    WI = PARS[0]
    EI = PARS[1]
    GI = PARS[2]
    nl = len(PARS[0,:])
    ns = len(xi)
    res = 0.0
    for i in range(nl):
        for j in range(i+1, nl):
            res += 2 * GI[i]*GI[j]*Overlap_Hilbert_Lrz(EI[i], WI[i], EI[j], WI[j])
        res +=     GI[i]*GI[i]*Overlap_Hilbert_Lrz(EI[i], WI[i], EI[i], WI[i])
    
    eps  = offset_E + .0 # make it agree with jacobian of error
    res2 = 0.0
    #Overlap_hat_HilbLrnz
    x1,x2,x3 = xi[0]-eps, xi[0], xi[1]
    for j in range(nl):
        res2 += Overlap_hat_HilbLrnz(x1, x2, x3, EI[j], WI[j]) * fi[0] * GI[j]
    
    x1,x2,x3 = (xi[ns-2], xi[ns-1], xi[ns-1] + eps)
    for j in range(nl):
        res2 += Overlap_hat_HilbLrnz(x1, x2, x3, EI[j], WI[j]) * fi[ns-1] * GI[j]
    
    for i in range(1,ns-1):
        x1,x2,x3 = xi[i-1:i+2]
        for j in range(nl):
            res2 += Overlap_hat_HilbLrnz(x1, x2, x3, EI[j], WI[j]) * fi[i] * GI[j]
    return res - 2*res2

# An extension of the above function to work on several entries of a matrix function

@njit(cache = CACHE, parallel = PARALLEL, fastmath = FASTMATH)
def delta_error_many_linear_hilbertt_lrz(xi, fiv, PARS_v):
    # xiv : ( n), sampling points
    # fiv : (ne, n), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    for i in prange(n):
        res += delta_error_linear_hilbertt_lrz(xi, fiv[:,i], PARS_v[:, :, i]).real
    return res
# Extension to complex values
@njit(cache = CACHE,parallel = PARALLEL, fastmath = FASTMATH)
def delta_error_many_linear_hilbertt_lrz_complex(xi, fiv, PARS_v):
    # xiv : ( n), sampling points
    # fiv : (ne, n), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    PARS_vi = np.zeros(PARS_v.shape)
    PARS_vi[0:2,:,:] = PARS_v[0:2, :,:].real
    PARS_vi[2,  :,:] = PARS_v[2  , :,:].imag
    
    for i in prange(n):
        res += delta_error_linear_hilbertt_lrz(xi, fiv[:,i].real, PARS_v [:, :, i].real).real + \
               delta_error_linear_hilbertt_lrz(xi, fiv[:,i].imag, PARS_vi[:, :, i].real).real
    return res
    
    
# An error function for a sum of Lorentzians and a sampled function f
# xi are the sampled points, 
# fi are the sampled values of x
# PARS is a (3,N) array containing the collection of linewidths (WI),
# the centres (EI) and the weights (GI).
# It returns the error of the fit (without the constant term)

@njit(cache = CACHE, fastmath = FASTMATH)
def delta_error_linear_lorentzian_old(xi, fi, PARS):
    WI = PARS[0]
    EI = PARS[1]
    GI = PARS[2]
    nl = len(PARS[0,:])
    ns = len(xi)
    res = 0.0
    for i in range(nl):
        for j in range(i+1, nl):
            res += 2 * GI[i]*GI[j]*Overlap(EI[i], WI[i], EI[j], WI[j])
        res +=     GI[i]*GI[i]*Overlap(EI[i], WI[i], EI[i], WI[i])
    
    eps  = offset_E + .0 # make it agree with jacobian of error
    res2 = 0.0
    
    x1,x2,x3 = xi[0]-eps, xi[0], xi[1]
    for j in range(nl):
        res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[0] * GI[j]
    
    x1,x2,x3 = (xi[ns-2], xi[ns-1], xi[ns-1] + eps)
    
    fcut_l = np.zeros(nl)
    for j in range(nl):
        res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[ns-1] * GI[j]
        fcut_l[j] = np.sqrt(GI[j]**2/Lrz_cutoff - GI[j]**2)
    
    dij = np.zeros((ns, nl))
    for i in range(ns):
        for j in range(nl):
            dij[i,j] = np.abs(EI[j] - xi[i])
    
    for i in range(1,ns-1):
        x1,x2,x3 = xi[i-1:i+2]
        for j in range(nl):
            dist  = dij[i-1:i+2,j].min()
            if dist<fcut_l[j]:
                res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[i] * GI[j]
    return res - 2 * res2

@njit(cache = CACHE, fastmath = FASTMATH)
def delta_error_linear_lorentzian(xi, fi, PARS, exc_idx = None):
    if exc_idx is None:
        return delta_error_linear_lorentzian_old(xi,fi,PARS)
    WI = PARS[0]
    EI = PARS[1]
    GI = PARS[2]
    nl = len(PARS[0,:])
    ns = len(xi)
    res= 0.0
    for i in range(nl):
        for j in range(i+1, nl):
            res += 2 * GI[i]*GI[j]*Overlap(EI[i], WI[i], EI[j], WI[j])
        res +=     GI[i]*GI[i]*Overlap(EI[i], WI[i], EI[i], WI[i])
    
    eps  = offset_E + .0 # make it agree with jacobian of error
    res2 = 0.0
    
    x1,x2,x3 = xi[0]-eps, xi[0], xi[1]
    for j in range(nl):
        if 0 not in exc_idx[j]:
            res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[0] * GI[j]
    
    x1,x2,x3 = (xi[ns-2], xi[ns-1], xi[ns-1] + eps)
    fcut_l   = np.zeros(nl)
    for j in range(nl):
        if (ns-1) not in exc_idx[j]:
            res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[ns-1] * GI[j]
        fcut_l[j] = np.sqrt(GI[j]**2/Lrz_cutoff - GI[j]**2)
    
    dij = np.zeros((ns, nl))
    for i in range(ns):
        for j in range(nl):
            dij[i,j] = np.abs(EI[j] - xi[i])
    # 
    # for i in range(1,ns-1):
    #     x1,x2,x3 = xi[i-1:i+2]
    #     for j in range(nl):
    #         dist  = dij[i-1:i+2,j].min()
    #         if dist<fcut_l[j]:
    #             res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[i] * GI[j]
    # 
    for j in range(nl):
        idx = npsetdiff1d(np.arange(1,ns-1), exc_idx[j])
        for i in idx:
            dist = dij[i-1:(i+2),j].min()
            x1,x2,x3 = xi[i-1:i+2]
            if dist<fcut_l[j]:
                res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[i] * GI[j]
    return res - 2 * res2

# Extension of the above function to matrix valued samplings f
@njit(cache = CACHE, parallel = PARALLEL, fastmath = FASTMATH)
def delta_error_many_linear_lorentzian(xi, fiv, PARS_v, exc_idx = None):
    # xiv :   (ne), sampling points
    # fiv :   (ne, nb), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    for i in prange(n):
        res += delta_error_linear_lorentzian(xi, fiv[:,i], PARS_v[:, :, i], exc_idx = exc_idx).real
    return res
    
# Extension to complex values
@njit(cache = CACHE,parallel = PARALLEL, fastmath = FASTMATH)
def delta_error_many_linear_lorentzian_complex(xi, fiv, PARS_v, exc_idx = None):
    # xiv : ( n), sampling points
    # fiv : (ne, n), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    PARS_vi = np.zeros(PARS_v.shape)
    PARS_vi[0:2,:,:] = PARS_v[0:2, :,:].real
    PARS_vi[2,  :,:] = PARS_v[2  , :,:].imag
    
    for i in prange(n):
        res += delta_error_linear_lorentzian(xi, fiv[:,i].real, PARS_v [:, :, i].real, exc_idx = exc_idx).real + \
               delta_error_linear_lorentzian(xi, fiv[:,i].imag, PARS_vi[:, :, i].real, exc_idx = exc_idx).real
    return res

# Constructs the NL x NL matrix containing the overlap integrals of the Lorentzians
@njit(cache = CACHE, fastmath = FASTMATH)
def lrz_overlap_matrix(e,w, compute_Oij=True):
    # O is normalised overlap
    # o is the numbers used to normalise
    n = len(e)
    o = np.zeros(n); O = np.zeros((n,n))
    for i in range(n):
        o[i] = Overlap(e[i],w[i],e[i],w[i])
    if compute_Oij:
        for i in range(n):
            for j in range(n):
                if i!=j:
                    O[i,j] = Overlap(e[i],w[i],e[j],w[j] ) / np.sqrt(o[i] * o[j])
    return O,o



# For the error on the hilbert transform of the Lorentzians
# Some integrals for 
#        Int x^[0,1,2] L(x, ei, wi) dx
# Wolfram alpha your way to these
 
@njit(cache = CACHE, fastmath = FASTMATH)
def Int2(x1, x2, a, b):
    lower = a*ln(a**2 - 2*a*x1 + b**2 + x1**2) + (b - a**2 / b) * atan((a-x1)/b) + x1
    upper = a*ln(a**2 - 2*a*x2 + b**2 + x2**2) + (b - a**2 / b) * atan((a-x2)/b) + x2
    return upper - lower

@njit(cache = CACHE, fastmath = FASTMATH)
def Int1(x1, x2, a, b):
    lower = 1/2 * ln(a**2 - 2*a*x1 + b**2 + x1**2) - a*atan((a - x1)/b)/b
    upper = 1/2 * ln(a**2 - 2*a*x2 + b**2 + x2**2) - a*atan((a - x2)/b)/b
    return upper - lower

@njit(cache = CACHE, fastmath = FASTMATH)
def Int0(x1, x2, a, b):
    lower = - atan((a - x1) / b) / b
    upper = - atan((a - x2) / b) / b
    return upper - lower

@njit(cache = CACHE, fastmath = FASTMATH)
def Overlap_hat_HilbLrnz(x1, x2, x3, ei, gi):
    dxm = x2-x1 
    dxp = x3-x2
    t1  = + gi        * Int2(x1,x2,ei,gi) \
          - x1  * gi  * Int1(x1,x2,ei,gi) \
          - ei  * gi  * Int1(x1,x2,ei,gi) \
          + gi*ei*x1  * Int0(x1,x2,ei,gi)
    
    t2  = - gi         * Int2(x2,x3,ei,gi) \
          + x3  * gi   * Int1(x2,x3,ei,gi) \
          + ei  * gi   * Int1(x2,x3,ei,gi) \
          - gi*ei*x3   * Int0(x2,x3,ei,gi)
    return t1/dxm + t2/dxp

# from scipy.integrate import quad
# X1,X2,X3 = 2.3, 3.34, 5.0
# Ei, Gi   = 3.0, 0.5

# def phi(x):
#     if X1<=x<X2:
#         return (x - X1)/(X2-X1)
#     if X2<=x<X3:
#         return (X3-x)/(X3-X2)
#     return 0.0

# def Hlrz(x):
#     return (Gi/( (x - Ei)**2 + Gi**2))*(x-Ei)

# def f(x):
#     return 1/((x - Ei )**2 + Gi**2)

# def prod(x):
#     return phi(x) * Hlrz(x)

# def f1(x):
#     return (x**2)*f(x)

# res1 = quad(prod, X1 - 2.5, X3 + 3.5)
# res2 = Overlap_hat_HilbLrnz(X1,X2,X3,Ei, Gi)

# A function that gives a very large value when the overlaps are too close
# 
@njit(cache = CACHE, fastmath = FASTMATH)
def Punish_overlap(e,w):
    O,o = lrz_overlap_matrix(e,w)
    return np.tan(O * np.pi/2).sum()

# enforcing O_ii = 1
@njit(cache =CACHE, fastmath = FASTMATH)
def d_normed_overlap_d_gamma(ei,wi,ej,wj):
    eps = 1e-5
    Oij_1 = Overlap(ei,wi + eps, ej, wj)
    Oij_2 = Overlap(ei,wi - eps, ej, wj)
    Oii_1 = Overlap(ei,wi+eps,ei,wi+eps)
    Oii_2 = Overlap(ei,wi-eps,ei,wi-eps)
    Ojj   = Overlap(ej,wj,ej,wj)
    return (Oij_1/np.sqrt(Oii_1*Ojj) - Oij_2/np.sqrt(Oii_2*Ojj))/(2*eps)


# Manual version for checking the gradient of Punish_overlap
@njit(cache = CACHE, fastmath = FASTMATH)
def manual_grad_Punish(e,w):
    assert 1 == 0
    eps = 1e-5
    n = len(e)
    grad = np.zeros((2, n))
    for i in range(n):
        de = np.zeros(n)
        de[i] = eps
        grad[0,i] = (Punish_overlap(e+de, w) - Punish_overlap(e-de,w))/(2*eps)
        grad[1,i] = (Punish_overlap(e, w+de) - Punish_overlap(e,w-de))/(2*eps)
    return grad

# Gradient of Punish_overlap
@njit(cache = CACHE, fastmath = FASTMATH)
def grad_Punish_overlap(e,w):
    n = len(e)
    grad = np.zeros((2, n))
    #print(e,w)
    
    O,o  = lrz_overlap_matrix(e, w)
    os   = np.sqrt(o)
        
    #print(os)
    #if (os<1e-12).any():
    #    print(e,w)
    for i in range(n):
        ei, wi = e[i], w[i]
        soii   = os[i]
        for j in range(n):
            ej, wj = e[j], w[j]
            sojj   = os[j]
            if i!=j:
                grad[0,i] += 2*d_tan_dx(O[i,j])*d_Overlap_d_eps         (ei,wi,ej,wj)/(soii*sojj)
                grad[1,i] += 2*d_tan_dx(O[i,j])*d_normed_overlap_d_gamma(ei,wi,ej,wj)
                
    return grad


# derivative of the punishing function Punish_overlap
@njit(cache = CACHE, fastmath = FASTMATH)
def d_tan_dx(x):
    return np.pi/2*(1/np.cos(x*np.pi/2))**2

# Gradient of the error of a sum of Lorentzians vs the linear interpolant f
# Only gradient in ei, wi is actually needed so you see some lines commented 
# for speed.

@njit(cache = CACHE, parallel = PARALLEL, fastmath = FASTMATH)
def jacobian_delta_error_linear_lorentzian(xi, fiv, ei, wi, Gi_v):
    nb   = len(fiv[0,:])
    res  = np.zeros(len(ei) + len(wi) + len(Gi_v[:,0]) * len(Gi_v[0,:]))
    n_g  = len(ei)
    nfi  = len(xi)
    it   = 0
    #eps  = 1.0 #make it agree with total error
    eps  = offset_E + .0
    idx1 = np.arange(0, n_g)
    idx2 = np.arange(n_g, 2 * n_g)
    idx3 = np.arange(2 * n_g, 2 * n_g + n_g * nb).reshape(n_g, nb)
    Oij        = np.zeros((n_g, n_g))
    dOijdeps   = np.zeros((n_g, n_g))
    dOijdgam   = np.zeros((n_g, n_g))
    hwl        = np.zeros((nfi, n_g))
    d_hwl_deps = np.zeros((nfi, n_g))
    d_hwl_dgam = np.zeros((nfi, n_g))
    
    for i in range(n_g):
        for j in range(n_g):
            Oij[i,j]      = Overlap(ei[i], wi[i], ei[j], wi[j])
            dOijdeps[i,j] = d_Overlap_d_eps  (ei[i], wi[i], ei[j], wi[j])
            dOijdgam[i,j] = d_Overlap_d_gamma(ei[i], wi[i], ei[j], wi[j])
        for j in range(nfi):
            if   j == 0:     x1 = xi[0]-eps;  x2 = xi[0];     x3 = xi[1]
            elif j == nfi-1: x1 = xi[nfi-2];  x2 = xi[nfi-1]; x3 = xi[nfi-1] + eps
            else:             x1,x2,x3 = xi[j-1:j+2]
            hwl       [j,i] = hat_with_lrtz          (x1,x2,x3, ei[i], wi[i])
            d_hwl_deps[j,i] = d_hat_with_lrtz_d_eps  (x1,x2,x3, ei[i], wi[i])
            d_hwl_dgam[j,i] = d_hat_with_lrtz_d_gamma(x1,x2,x3, ei[i], wi[i])
    
    for i in prange(n_g):
        for b in range(nb):
            for m in range(n_g):
                res[idx1[i]] += 2 * Gi_v[m,b] * Gi_v[i,b] * dOijdeps[i,m]#d_Overlap_d_eps(ei[i], wi[i], ei[m], wi[m])
            for ii in range(nfi):
                res[idx1[i]] -= 2 * fiv[ii, b] * Gi_v[i,b] * d_hwl_deps[ii,i]#d_hat_with_lrtz_d_eps(x1,x2,x3, ei[i], wi[i])
        for b in range(nb):
            for m in range(n_g):
                if m==i:
                    res[idx2[i]] += Gi_v[m,b] * Gi_v[i,b] * d_Overlap_d_gamma_same(ei[i], wi[i])
                else:
                    res[idx2[i]] += 2 * Gi_v[m,b] * Gi_v[i,b] * dOijdgam[i,m]#d_Overlap_d_gamma(ei[i], wi[i], ei[m], wi[m])
            for ii in range(nfi):
                res[idx2[i]] -= 2 * fiv[ii, b] * Gi_v[i,b] * d_hwl_dgam[ii,i]#d_hat_with_lrtz_d_gamma(x1,x2,x3, ei[i], wi[i])
    return res

#Complex extension of the previous function
@njit(cache = CACHE, fastmath = FASTMATH)
def jacobian_delta_error_linear_lorentzian_complex(xi, fiv, ei, wi, Gi_v):
    fiv_i = fiv.imag
    fiv_r = fiv.real
    nl    = len(ei)
    nb    = len(fiv[0,:])
    jac   = np.zeros(2 *nl + nl * nb*2)
    
    Gi_vi = Gi_v.imag
    Gi_vr = Gi_v.real
    jac_r = jacobian_delta_error_linear_lorentzian(xi, fiv_r, ei, wi, Gi_vr)
    jac_i = jacobian_delta_error_linear_lorentzian(xi, fiv_i, ei, wi, Gi_vi)
    
    jac[0:nl]                     = jac_r[0:nl]    + jac_i[0:nl]
    jac[nl:2*nl]                  = jac_r[nl:2*nl] + jac_i[nl:2*nl]
    jac[2*nl:2*nl+nb*nl]          = jac_r[2*nl:2*nl+nb*nl] 
    jac[2*nl+nb*nl:2*nl+2*nb*nl]  = jac_i[2*nl:2*nl+nb*nl]
    
    return jac

# Function for checking
def ez_wrap(xi, fiv, ei, wi, Gi_v):
    PARS = np.zeros((3,)+ei.shape+fiv[0,:].shape, dtype = np.complex128)
    PARS[0,:,:] = wi[:, np.newaxis]
    PARS[1,:,:] = ei[:, np.newaxis]
    PARS[2,:,:] = Gi_v
    return delta_error_many_linear_lorentzian_complex(xi, fiv, PARS)

# def L1L2(x):
#     return L(x , 0.01, -0.0) * L(x , 0.01001, 0)


#Shorthand for making a sum of Lorentzians
@njit(cache = CACHE, fastmath = FASTMATH)
def L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0,:])
    res = np.zeros(len(E), dtype = np.complex128)
    
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l])
    return res

# Find the best fit some function f sampled in zi, when you constrain the values of ei and wi (gamma in the function)
@njit(cache = CACHE, fastmath = FASTMATH)
def Lorentzian_basis(f, zi, ei, gamma):
    assert len(zi) == len(ei) == len(gamma)
    nl =  len(zi)
    B  =  np.zeros((nl,nl), dtype = np.complex128)
    for i in range(nl):
        for j in range(nl):
            B[i,j] = L(zi[i], gamma[j], ei[j])
    return np.linalg.inv(B).dot(f)


# Convert matrix of sampled values to an interpolated matrix defined from the centers are linewidths ei and gamma(in the function)
@njit(cache = CACHE, fastmath = FASTMATH)
def old_Matrix_lorentzian_basis(M, zi, ei, gamma):
    res = np.zeros(M.shape, dtype = M.dtype)
    #print(zi.shape, ei.shape, gamma.shape)
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                res[ik, :, i, j] = Lorentzian_basis(M[ik, :, i, j], zi, ei[ik], gamma[ik])
    return res
# This function has replaced the above function on 22/06/2022 because the previous
# caused singular matrices sometimes. Was only used in initialisation before
# 15.12.22 added tol keyword
@njit(cache = CACHE, fastmath = FASTMATH)
def Matrix_lorentzian_basis(M, zi, ei, gamma, tol=1e-15, exc_idx = None):
    res = np.zeros(M.shape, dtype = M.dtype)
    #print(zi.shape, ei.shape, gamma.shape)
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    for ik in range(nk):
        lm = gamma_from_centres_matrix(ei[ik],gamma[ik])
        ilm= np.linalg.inv(lm)
        for i in range(ni):
            for j in range(nj):
                if (np.abs(M[ik,:,i,j])>tol).any():
                    res[ik, :, i, j] = ilm.dot(gamma_from_centers_RHS(ei[ik],gamma[ik],M[ik,:,i,j], zi, exc_idx = exc_idx))
    return res

# Evaluate the values of a Lorentzian expansion on an energy grid
# 15.12.22 added tol keyword
@njit(cache = CACHE, fastmath = FASTMATH)
def evaluate_Lorentz_basis_matrix(M, E, ei, gamma, tol = 1e-15):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni, nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                if (np.abs(M[ik,:,i,j])>tol).any():
                    pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                    res[ik,:,i,j] = L_sum(E, pars)
    return res

@njit(cache = CACHE, fastmath = FASTMATH)
def evaluate_Lorentz_basis_matrix_hermitian(M, E, ei, gamma, tol = 1e-15):
    nk  = M.shape[0]
    ni  = M.shape[2]
    nj  = M.shape[3]
    res = np.zeros((nk, len(E), ni, nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(i,nj):
                if (np.abs(M[ik,:,i,j])>tol).any():
                    pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                    res[ik,:,i,j] = L_sum(E, pars)
                    if i!=j:
                        res[ik,:,j,i] = res[ik,:,i,j].conj()   
    return res

# 15.12.22 added tol keyword
@njit(cache = CACHE, fastmath = FASTMATH)
def evaluate_KK_matrix(M, E, ei, gamma, tol = 1e-15):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni,nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                if (np.abs(M[ik,:,i,j])>tol).any():
                    pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                    res[ik,:,i,j] = KK_L_sum(E, pars)
    return res

@njit(cache = CACHE, fastmath = FASTMATH)
def evaluate_KK_matrix_hermitian(M, E, ei, gamma, tol = 1e-15):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni,nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(i,nj):
                if (np.abs(M[ik,:,i,j])>tol).any():
                    pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                    res[ik,:,i,j] = KK_L_sum(E, pars)
                    if i!=j:
                        res[ik,:,j,i] = res[ik,:,i,j].conj()
    return res

#Evaluate the hilbert transform of the Lorentzians on a energy grid
@njit(cache = CACHE, fastmath = FASTMATH)
def KK_L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0,:])
    res = np.zeros(len(E), dtype = np.complex128)
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l]) * (E - ei[l]) / Wi[l]
    return res

@njit(cache = CACHE, fastmath = FASTMATH)
def gamma_from_centres_matrix(ei,wi):
    n = len(ei)
    Mat = np.zeros((n,n),dtype = np.complex128)
    for i in range(n):
        for j in range(n):
            Mat[i,j] = Overlap(ei[i], wi[i], ei[j], wi[j])
    return Mat

@njit(cache = CACHE, fastmath = FASTMATH)
def gamma_from_centers_RHS_old(ei, wi, fi, xi):
    n = len(ei)
    nf = len(fi)
    v   = np.zeros(n, dtype = np.complex128)
    for i in range(n):
        eii, gii  = ei[i], wi[i]
        dists     = np.abs(eii - xi)
        fcut      = np.sqrt(gii**2/Lrz_cutoff - gii**2)
        for j in range(nf):
            fij   = fi[j]
            if j == 0:
                x1,x2,x3 = xi[j]-offset_E,xi[j],xi[j+1]
                d1,d2,d3 = np.abs(eii - x1), dists[j], dists[j+1]
            elif j == nf-1:
                x1,x2,x3 = xi[j-1],xi[j],xi[j]+offset_E
                d1,d2,d3 = dists[j-1], dists[j], np.abs(eii - x3)
            else:
                x1,x2,x3 = xi[j-1],xi[j],xi[j+1]
                d1,d2,d3 = dists[j-1], dists[j], dists[j+1]
            if min((d1,d2,d3))<fcut:
                v[i] += fij * hat_with_lrtz (x1,x2,x3,eii,gii)
    return v

@njit(cache = CACHE, fastmath = FASTMATH)
def gamma_from_centers_RHS(ei, wi, fi, xi, exc_idx = None ):
    if exc_idx is None:
        return gamma_from_centers_RHS_old(ei,wi,fi,xi)
    n   = len(ei)
    nf  = len(fi)
    v   = np.zeros(n, dtype = np.complex128)
    for i in range(n):
        eii, gii  = ei[i], wi[i]
        dists     = np.abs(eii - xi)
        fcut      = np.sqrt(gii**2/Lrz_cutoff - gii**2)
        exc       = exc_idx[i]
        idx       = npsetdiff1d(np.arange(nf), exc)
        for j in idx:
            fij = fi[j]
            if j == 0:
                x1,x2,x3 = xi[j]-offset_E,xi[j],xi[j+1]
                d1,d2,d3 = np.abs(eii - x1), dists[j], dists[j+1]
            elif j == nf-1:
                x1,x2,x3 = xi[j-1],xi[j],xi[j]+offset_E
                d1,d2,d3 = dists[j-1], dists[j], np.abs(eii - x3)
            else:
                x1,x2,x3 = xi[j-1],xi[j],xi[j+1]
                d1,d2,d3 = dists[j-1], dists[j], dists[j+1]
            if min((d1,d2,d3))<fcut:
                v[i] += fij * hat_with_lrtz (x1,x2,x3,eii,gii)
    return v


#@njit(cache = True)
def gammas_from_centers_vec(ei,wi,fiv,xi, exc_idx = None):
    nl = len(ei)
    nb = len(fiv[0,:])
    Gi = np.zeros((nl,nb), dtype = np.complex128)
    Mat = gamma_from_centres_matrix(ei,wi)
    
    try:
        iM=np.linalg.inv(Mat)
        for b in range(nb):
            Gi[:,b] = iM.dot(gamma_from_centers_RHS(ei,wi,fiv[:,b], xi,exc_idx = exc_idx))
    except:
        print('BAD LHS: Using '+ Gamma_fallback)
        if Gamma_fallback == 'random':
            Mat = gamma_from_centres_matrix(ei + np.random.random(nl)*0.01,wi)
            iM = np.linalg.inv(Mat)
            for b in range(nb):
                Gi[:,b] = np.dot(iM, gamma_from_centers_RHS(ei,wi,fiv[:,b], xi, exc_idx = exc_idx))
        elif Gamma_fallback == 'pseudo-inverse':
            Mat = gamma_from_centres_matrix(ei,wi)
            iM = np.linalg.pinv(Mat)
            for b in range(nb):
                Gi[:,b] = np.dot(iM, gamma_from_centers_RHS(ei,wi,fiv[:,b], xi, exc_idx = exc_idx))
    if Gamma_default == 'Inversion':
        pass
    elif Gamma_default == 'Curvefit':
        assert 1 == 0
    return Gi

def build_matrix(Gi, idxi,idxj, n):
    nl, nb = Gi.shape
    assert len(idxi) == nb
    assert len(idxj) == nb
    
    M = np.zeros((nl, n,n), dtype = Gi.dtype)
    M[:,idxi, idxj] = Gi[:,:]
    return M

def build_raveled(M, idxi, idxj):
    return M[:,idxi,idxj]

@njit(cache =CACHE, fastmath = FASTMATH)
def make_hermitian_single(m, tol = 1e-10):
    n = len(m)
    M = m.copy()
    for i in range(n):
        for j in range(i+1,n):
            if abs(M[i,j] - np.conj(M[j,i]))<tol:
                pass
            elif abs(M[i,j])>tol and  abs(M[j,i])<tol:
                M[j,i]  = np.conj(M[i,j])
            elif abs(M[j,i])>tol and  abs(M[i,j])<tol:
                M[i,j]  = np.conj(M[j,i])
    return M

@njit(cache = CACHE, fastmath = FASTMATH)
def make_hermitian(M,tol = 1e-10):
    out = np.zeros(M.shape,dtype = M.dtype)
    n = len(M)
    for i in range(n):
        out[i] = make_hermitian_single(M[i], tol = tol)
    return out

def force_PSD_func(Gi, idxi,idxj,n, hermitian = True, min_tol = 0.0, set_to_zero = False, ei =  None, wi = None):
    M = build_matrix(Gi, idxi, idxj, n)
    if hermitian:
        M = make_hermitian(M)
        eigsol = np.linalg.eigh
    else:
        eigsol = np.linalg.eig
    e, v = eigsol(M)
    
    if isinstance(min_tol, float):
        if set_to_zero:
            e[e.real<min_tol] = 0.0
        elif np.isnan(min_tol):
            
            if (ei is not None and wi is not None):
                #(M.)
                compute_new_eigs_2(M, ei, wi,)
                return build_raveled(M, idxi, idxj)
            else:
                return build_raveled(M, idxi, idxj)
        else:
            e[e.real<min_tol] = min_tol
    elif isinstance(min_tol, np.ndarray):
        assert len(min_tol.shape) == 1
        if set_to_zero:
            e[e.real<min_tol[:,None]] = 0.0
        else:
            e[e.real<min_tol[:, None]] = np.repeat(min_tol[:, None], 
                                                   len(e[0,:]),
                                                   axis=1)[e.real<min_tol[:, None]]
    if hermitian:
        out =  v@(e[... , None]*v.conj().transpose(0,2,1))
    else:
        out =  v@(e[... , None]*np.linalg.inv(v))
    #print('Out: ', e)
    
    return build_raveled(out, idxi, idxj)

@njit
def minaxis1(a):
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        out[i] = np.min(a[i,:])
    return out

        
    


#@njit(fastmath=True)
# def compute_new_eigs_for_PSD(_e,ei,wi, n=5, nW = 5, eps = 0.001, fact= 0.75, maxit = 500, lbtol = 0.0, vecs = None):
#     if len(_e.shape)==3:
#         e       = compute_new_eigs_2(_e, ei, wi, n=n, nW=nW, eps=eps, fact=fact, maxit=maxit, lbtol=lbtol)
#         _e[:,:,:] = e[:,:,:]
#         return
    
#     if (_e.real>=0.0).all(): return
#     e      = _e.copy()
#     nl,neig= len(ei), e.shape[1]
#     ng     = 2 * n + 1
#     Eg     = np.zeros(ng * nl,dtype=np.complex128)
#     points = np.linspace(-nW, nW, ng)
#     #print(points)
#     for i in range(nl):
#         Eg[i * ng : (i+1)*ng] = points*wi[i] + ei[i]
#     #A is the lower bound for the eigenvalues of the combined fit.
#     c     = np.vstack((wi, ei, minaxis1(e.real).astype(np.complex128)))
#     A     = L_sum(Eg, c)
#     lsamp = np.zeros(e.shape,dtype = np.complex128)
#     it    = 0
    
#     while A.min().real<lbtol and it < maxit:
#         c      = np.vstack((wi, ei, minaxis1(e.real)))
#         A      = L_sum(Eg, c)
#         I_Amin = np.where(A == A.min())
#         eAmin  = Eg[I_Amin[0]][0]
#         for i in range(nl):
#             Li = L(eAmin, wi[i], ei[i])
#             for j in range(neig):
#                 # print(Li.shape)
#                 lsamp[i,j] = Li * e[i,j]
        
#         I_lmin, J_lmin = np.where(lsamp == lsamp.min())
#         if A.min().real<lbtol:
#             e[I_lmin[0], J_lmin[0]] = (e[I_lmin[0], J_lmin[0]] + eps)*fact
#         it += 1
#     print(it)
#     idx = np.argsort(Eg)
#     plt.plot(Eg[idx].real,A[idx].real,label = 'A')
#     _e[:,:] = e[:,:]
#     return

@njit(fastmath = FASTMATH, cache=CACHE)
def compute_new_eigs_2(_M, ei,wi,n=DE_n, nW=DE_nW, eps=DE_eps, fact = DE_fact, maxit=DE_maxit, lbtol=DE_lbtol,
                       add_last = False):
    print('Refining eigenvalues of Fit!')
    M    = _M.astype(np.complex128)
    E, V = np.zeros(_M.shape[0:2], dtype=np.complex128),np.zeros(_M.shape, dtype=np.complex128)  
    for i in range(E.shape[0]):
        E[i], V[i] = np.linalg.eigh(_M[i])
    
    nl,neig = len(ei), M.shape[2]
    ng     = 2 * n + 1
    Eg     = np.zeros(ng * nl,dtype=np.complex128)
    points = np.linspace(-nW, nW, ng)
    Eg     = np.zeros((ng * nl),dtype=np.complex128)
    ev     = np.zeros((ng*nl))
    A      = np.zeros((neig,neig),   dtype=np.complex128)
    Lvec   = np.zeros((ng * nl, nl), dtype=np.complex128)
    for i in range(nl):
        Eg[i * ng : (i+1)*ng] = points*wi[i] + ei[i]
    for i in range(ng*nl):
        A[:,:] = 0.0
        for l in range(nl):
            A += M[l] * L(Eg[i], wi[l], ei[l])
        e   = np.linalg.eigvalsh(A)
        ev[i] = e.min()
    lsamp = np.zeros(E.shape,dtype = np.complex128)
    it    = 0
    for i in range(ng*nl):
        for l in range(nl):
            Lvec[i,l] = L(Eg[i], wi[l], ei[l])
    idx = np.arange(ng*nl)
    while ev.min().real<lbtol and it<maxit:
        computeA(M, Lvec, idx, Eg, ev, nl, neig)
        I_Amin = np.where(ev == ev.min())
        eAmin  = Eg[I_Amin[0]][0]
        for i in range(nl):
            Li         = L(eAmin, wi[i], ei[i])
            for j in range(neig):
                lsamp[i,j] = Li * E[i,j] #* np.abs(np.dot(vmin[:,0].conj(), V[i, :, j]))**2
        I_lmin, J_lmin = np.where(lsamp == lsamp.min())
        if ev.min().real<lbtol:
            #original value
            eor = E[I_lmin[0], J_lmin[0]]
            # shrunk value
            E[I_lmin[0], J_lmin[0]] = E[I_lmin[0], J_lmin[0]] *fact + eps
            # add change to matrix
            val = (E[I_lmin[0], J_lmin[0]] - eor)
            vec = V[I_lmin[0], :, J_lmin[0]]
            add_outer(vec, val, M[I_lmin[0]])
        imin = I_lmin[0]
        if np.mod(it,5) == 0:
            idx  = np.where(np.abs(Eg - ei[imin])<(10*wi[imin].real))[0]
            if np.mod(it,10) == 0:
                print('refinement pass '+ str(it) + ',  minimum eigenvalue: ', ev.min())
        else:
            idx  = np.where(np.abs(Eg - ei[imin])<(10*wi[imin].real))[0]
        it += 1
    if it == maxit:
        print('Calculation didnt finish trim of eigenvalues!')
    print('it, mineigv: ', it, ev.min())
    Id = np.eye(neig)
    for i in range(nl):
        if E[i].real.min()<0.0 and add_last:
            M[i] += abs(lbtol) * Id
    return M

@njit(fastmath = FASTMATH, parallel = PARALLEL, cache=CACHE)
def computeA(M, Lvec, idx, Eg, ev, nl, neig):
    for k in prange(len(idx)):
        i = idx[k]
        A = np.dot(M.reshape(nl, neig**2).T, Lvec[i]).reshape(neig,neig)
        if check_PSD(A):
            e = np.zeros(neig)
        else:
            e = np.linalg.eigvalsh(A)
        ev[i] = e.min()
    
    return
    # idx = np.argsort(Eg)
    # plt.plot(Eg[idx].real,ev[idx].real+0.01,label = 'A')

@njit(fastmath=True, cache=CACHE)
def add_outer(v,f, out):
    n  = len(v)
    sf = f**0.5
    V  = v*sf
    Vc = np.conj(V)
    for i in range(n):
        for j in range(n):
            out[i,j] += V[i]*Vc[j]

@njit(cache=CACHE)
def check_PSD(A):
    try:
        np.linalg.cholesky(A)
        return True
    except:
        return False

@njit
def slowcheck_PSD(A):
    e = np.linalg.eigvalsh(A)
    if (e>=0).all():
        return True
    return False



# @njit
# def HermGershgorin(A):
#     #"""Returns minimum bound on the eigenvalues of the matrix A"""
#     n = len(A)
#     v = np.zeros(n, dtype=np.complex128)
#     for i in range(n):
#         for j in range(i):
#             v[i]+= np.abs(A[i,j])
#         for i in range(i+1,n):
#             v[i]+= np.abs(A[i,j])
#         v[i] = A[i,i] - v[i]
#     return v

# def compare(M):
#     print(np.linalg.eigvalsh(M).min())
#     print(HermGershgorin(M).min())


# import matplotlib.pyplot as plt
# # # from Zandpack.LanczosAlg import Lanczos
# ei, wi = np.linspace(-4,4,31).astype(complex), (np.ones(31)*0.5).astype(complex)
# ei    += np.random.random(31)*1e-8
# wi    += np.random.random(31)*1e-8

# #wi[::3] *= 0.5

# np.random.seed(22)
# N        = 50
# M        = np.random.random((31,N,N)).astype(complex)
# M       += M.transpose(0,2,1)
# e,v      = np.linalg.eigh(M)
# e[e<0]   =  0.
# e[10:12,:] = -1.5
# M   = v@(e[...,None] * (v.transpose(0,2,1).conj()))
# e,v = np.linalg.eigh(M)

# e        = e.astype(complex)
# eo       = e.copy()
# ed       = eo.copy()
# ed[ed<-0.0] = -0.00
# #compute_new_eigs_for_PSD(M, ei, wi, maxit=500, eps=0.01, fact = 0.25, lbtol=-0.0)

# MM   = v@(e[..., None]*v.conj().transpose(0,2,1))
# MMM  = v@(ed[..., None]*v.conj().transpose(0,2,1))
# MMMM = compute_new_eigs_2(M, ei, wi, maxit = 5000, fact = 0.05, eps=0.01, nW=3, n=3, lbtol=0.0)
# Eg   = np.linspace(-5,5,3001)
# ME1  = np.zeros((3001, N,N),dtype=complex)
# ME2  = np.zeros((3001, N,N),dtype=complex)
# ME3  = np.zeros((3001, N,N),dtype=complex)
# for i in range(3001):
#     for l in range(31):
#         ME1[i,:,:] += L(Eg[i], wi[l], ei[l])*MMMM[l]
#         ME2[i,:,:] += L(Eg[i], wi[l], ei[l])*MM[l]
#         ME3[i,:,:] += L(Eg[i], wi[l], ei[l])*MMM[l]

# e1 = np.linalg.eigvalsh(ME1)
# e2 = np.linalg.eigvalsh(ME2)
# e3 = np.linalg.eigvalsh(ME3)

# plt.plot(Eg,e1.min(axis=1).real, label = 'Alg')
# plt.plot(Eg,e2.min(axis=1).real, label = 'Raw')
# plt.plot(Eg,e3.min(axis=1).real, label = 'Brute force')
# plt.legend()
# print('Alg min: ', e1.min() )
# print('Raw min: ', e2.min() )
# print('Brute force min: ', e3.min() )



# plt.ylim(0,0.2)
# plt.ylim(-0.5,None)
# a,b,v = Lanczos(MM, N=30)
# M2 = np.diag(a[0])
# for i in range(len(M2)-1):
#     M2[i,i+1] = b[0,i]
#     M2[i+1,i] = b[0,i]
# eig2,vec2 = np.linalg.eigh(M2)

@njit#('int32[:](int32[::1], int32[::1])')
def npsetdiff1d(arr1,arr2):#setdiff1d_nb_simd(arr1, arr2):
    # THANKS https://python.tutorialink.com/the-most-efficient-way-rather-than-using-np-setdiff1d-and-np-in1d-to-remove-common-values-of-1d-arrays-with-unique-values/
    out = np.empty_like(arr1)
    limit = arr1.size // 4 * 4
    limit2 = arr2.size // 2 * 2
    cur = 0
    z32 = np.int32(0)

    # Tile (x4) based computation
    for i in range(0, limit, 4):
        f0, f1, f2, f3 = z32, z32, z32, z32
        v0, v1, v2, v3 = arr1[i], arr1[i+1], arr1[i+2], arr1[i+3]
        # Unrolled (x2) loop searching for a match in `arr2`
        for j in range(0, limit2, 2):
            val1 = arr2[j]
            val2 = arr2[j+1]
            f0 += (v0 == val1) + (v0 == val2)
            f1 += (v1 == val1) + (v1 == val2)
            f2 += (v2 == val1) + (v2 == val2)
            f3 += (v3 == val1) + (v3 == val2)
        # Remainder of the previous loop
        if limit2 != arr2.size:
            val = arr2[arr2.size-1]
            f0 += v0 == val
            f1 += v1 == val
            f2 += v2 == val
            f3 += v3 == val
        if f0 == 0: out[cur] = arr1[i+0]; cur += 1
        if f1 == 0: out[cur] = arr1[i+1]; cur += 1
        if f2 == 0: out[cur] = arr1[i+2]; cur += 1
        if f3 == 0: out[cur] = arr1[i+3]; cur += 1

    # Remainder
    for i in range(limit, arr1.size):
        if arr1[i] not in arr2:
            out[cur] = arr1[i]
            cur += 1

    return out[:cur]




# which_ei = 15
# xi = np.linspace(-2,2,100)
# ei = np.linspace(-1.5, 1.5, 20)
# wi = np.random.random(20)
# Gi_v = np.random.random((20, 6))+1j * np.random.random((20, 6))
# fiv = np.array([np.sin(xi) + 1j * np.cos(xi)]*6).T
# de = np.zeros(20); de[which_ei] = 1e-6
# dw = np.zeros(20); dw[which_ei] = 1e-6

# dg = np.zeros(Gi_v.shape,dtype = np.complex128); dg[0, 2] =  1j*1e-6
# g1 = (ez_wrap(xi, fiv, ei+de, wi, Gi_v) - ez_wrap(xi, fiv, ei-de, wi, Gi_v)) / 2e-6
# b1 = (ez_wrap(xi, fiv, ei, wi+dw, Gi_v) - ez_wrap(xi, fiv, ei, wi-dw, Gi_v)) / 2e-6
# h1 = (ez_wrap(xi, fiv, ei, wi, Gi_v + dg) - ez_wrap(xi, fiv, ei, wi, Gi_v - dg)) / 2e-6
# g2 = jacobian_delta_error_linear_lorentzian_complex(xi, fiv, ei, wi, Gi_v)
# print(g2[which_ei], g1)

#Gi_v2 = gammas_from_centers_vec(ei,wi,fiv,xi)
# import matplotlib.pyplot as plt

# E = np.linspace(-10,10, 1000)
# n = 10
# x1,c1 = Pade_poles_and_coeffs(n)
# x2,c2 = Hu_poles(n)
# plt.plot(E, FD(E, 1/0.025), label = 'Fermi function')
# plt.plot(E, FD_expanded(E, x1, 1/0.025,  coeffs = c1), label = 'Croy &  Saalman')
# plt.plot(E, FD_expanded(E, x2, 1/0.025,  coeffs = c2), label = 'Jie Hu 2011')
# plt.title('10 Poles')
# plt.legend()

# nl = 50
# x = np.linspace(-10,10, nl)  #np.array(sorted(np.random.random(50) * 5))
# f =  1/x #np.sin(x)
#f[np.where(f<0)] = 0

# xl = np.linspace(x.min(), x.max(), nl)
# g = 0.8 * (xl[1] - xl[0])
# M = np.zeros((2,4,3,nl))
# M[:,:,:,:] = f
# M = M.transpose(0,3,1,2)

# a = Lorentzian_basis(f.astype(np.complex128), x, xl, g)#np.linalg.solve(B,f)
# p = np.array([np.ones(nl)*g, xl, a])


# XX = np.linspace(x.min(), x.max(), 1000)
# ff = L_sum(XX, p)

# plt.plot(x,f.real)
# plt.plot(XX,ff.real)

# @njit
# def other_L(E, W, ei):
#     return Lp(E, W, ei) + Lm(E,W,ei)
# @njit
# def Lp(E,W,ei):
#     return -0.5j*W/(E - ei - 1j*W)
# @njit
# def Lm(E,W,ei):
#     return +0.5j*W/(E - ei + 1j*W)
# def optimizeweight(PARS,  wi, ei, all_terms=True):
#     def f(x):
#         return fit_error(PARS, wi, ei,x[0], all_terms = all_terms)
#     x0 = np.array(1.0)
#     sol = minimize(f, x0)
#     return sol.x

# def Matrix_optimize1L(Gi_M, Ei, Wi, emin, emax, min_width, max_width,all_terms = False):
#     nk = Gi_M.shape[0]
#     ni = Gi_M.shape[2]
#     nj = Gi_M.shape[3]
#     res = np.zeros((nk, ni, nj, 4))
    
#     for ik in range(nk):
#         for i in range(ni):
#             for j in range(nj):
#                 PARS = np.vstack((Wi[ik],Ei[ik],Gi_M[ik,:,i,j]))
#                 out = optimize1L(PARS, all_terms = all_terms,
#                                  min_width = min_width,
#                                  max_width = max_width, 
#                                  emin      = emin,
#                                  emax      = emax)
                
#                 res[ik,i,j, 0:3] = out
#                 res[ik,i,j,   3] = fit_error(PARS, out[0],out[1], out[2], all_terms = True)
    
#     return res

# def Matrix_optimizeweight(Gi_M, Ei, Wi, ei, wi, all_terms = False):
#     nk = Gi_M.shape[0]
#     ni = Gi_M.shape[2]
#     nj = Gi_M.shape[3]
#     res = np.zeros((nk, ni, nj,1))
    
#     for ik in range(nk):
#         for i in range(ni):
#             for j in range(nj):
#                 PARS = np.vstack((Wi[ik],Ei[ik],Gi_M[ik,:,i,j]))
#                 out = optimizeweight(PARS, wi[ik], ei[ik], 
#                                      all_terms = all_terms)
#                 res[ik,i,j] = out
                
#     return res

# Re = np.real
# @njit(cache = True)
# def delta_error_linear_lorentzian_complex(xi, fi, PARS):
#     WI = PARS[0]
#     EI = PARS[1]
#     GI = PARS[2]
#     GIc = GI.conj()
#     nl = len(PARS[0,:])
#     ns = len(xi)
#     res = 0.0
#     for i in range(nl):
#         for j in range(i+1, nl):
#             res += 2 * Re(GI[i]*GIc[j])*Overlap(EI[i], WI[i], EI[j], WI[j])
#         res +=     GI[i]*GIc[i]*Overlap(EI[i], WI[i], EI[i], WI[i])
    
#     eps  = 1.0 # make it agree with jacobian of error
#     res2 = 0.0
    
#     x1,x2,x3 = xi[0]-eps, xi[0], xi[1]
#     for j in range(nl):
#         res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * Re(fi[0] * GIc[j])
    
#     x1,x2,x3 = (xi[ns-2], xi[ns-1], xi[ns-1] + eps)
#     for j in range(nl):
#         res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * Re(fi[ns-1] * GIc[j])
    
#     for i in range(1,ns-1):
#         x1,x2,x3 = xi[i-1:i+2]
#         for j in range(nl):
#             res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * Re(fi[i] * GIc[j])
    
#     return res - 2 * res2
# @njit
# def d_Overlap_d_eps(ei,wi,ej,wj):
#     xi = ei + 1j * wi
#     xj = ej + 1j * wj
#     if abs(xi -  xj)< 1e-12:
#         res = 0.0
#     else:
#         res = 2j * np.pi/4 * wi * wj *( +1/(xj-np.conj(xi))**2 - 1/(xi - np.conj(xj))**2 + 1/(xi - xj)**2 - 1/(xj - xi)**2)
#     return res.real

# @njit
# def fit_error(PARS, wi, ei, gi, all_terms = True):
#     WI = PARS[0]
#     EI = PARS[1]
#     GI = PARS[2]
#     nl = len(PARS[0,:])
#     res = 0.0
#     if all_terms == True:
#         for i in range(nl):
#             for j in range(nl):
#                 res += GI[i]*GI[j]*Overlap(EI[i], WI[i], EI[j], WI[j])
    
#     res += gi ** 2 * Overlap(ei, wi, ei, wi)
#     for i in range(nl):
#         res -= 2 * gi * GI[i] * Overlap(ei, wi, EI[i], WI[i])
#     return res

# @njit
# def fit_error_sums(PAR1, PAR2, all_terms = True):
#     WI1 = PAR1[0]
#     EI1 = PAR1[1]
#     GI1 = PAR1[2]
#     nl1 = len(PAR1[0,:])
#     WI2 = PAR2[0]
#     EI2 = PAR2[1]
#     GI2 = PAR2[2]
#     nl2 = len(PAR2[0,:])
    
#     res = 0.0
    
#     if all_terms == True:
#         for i in range(nl1):
#             for j in range(nl1):
#                 res += GI1[i]*GI1[j]*Overlap(EI1[i], WI1[i], EI1[j], WI1[j])
#     for i in range(nl2):
#         for j in range(nl2):
#             res += GI2[i]*GI2[j]*Overlap(EI2[i], WI2[i], EI2[j], WI2[j])
#     for i in range(nl1):
#         for j in range(nl2):
#             res -= 2 * GI1[i] * GI2[j] *Overlap(EI1[i], WI1[i], EI2[j], WI2[j])
#     return res

# def optimize1L(PARS, all_terms=True,min_width = 0.1, max_width = 10.0, emin = -10.0, emax = 10.0 ):
#     def f(x):
        
#         return fit_error(PARS, x[0], x[1],x[2], all_terms = all_terms)/x[2]
    
#     x0 = np.array([min_width * 5, emin + (emax - emin) * np.random.random(), np.average(PARS[2])])
#     sol = minimize(f, x0, bounds= ((min_width,max_width), (emin , emax), (-100, 100)))
#     return sol.x

# def Matrix_fiterror_sum(Gi_M, Ei, Wi, Ei, kidx = 0):
# def Matrix_sumfit_error(x, Gi_M, Ei, Wi, all_terms = False, ik = 0):
#     assert x.shape[0] == Gi_M[0]
#     assert x.shape[2] == Gi_M[2]
#     assert x.shape[3] == Gi_M[3]
    
#     nk = Gi_M.shape[0]
#     ni = Gi_M.shape[2]
#     nj = Gi_M.shape[3]
#     res = np.zeros((nk, ni, nj, 4))
#     NL = len(x)//3
    
#     def f(x):
#         X = x.reshape(3,NL)
#         return fit_error_sums(PARS, X, all_terms = all_terms)
#     out = 0.0
    
#     for i in range(ni):
#         for j in range(nj):
#             PARS = np.vstack((Wi[ik],Ei[ik],Gi_M[ik,:,i,j]))
#             out += f(x[ik,:,i,j])
    
#     return out






# def PSD_fermi(N):
#     M = np.diag(np.zeros(2 * N))
    
    
#     for i in range(2 * N):
#         for j in range(2 * N):
#             I = i + 1
#             J = j + 1
            
#             if i == j +1 or j == i+1: 
#                 M[i,j] = 1/(((2 * J -1 ) * (2 * I - 1)) ** 0.5)
#     e,v = np.linalg.eig(M)
#     e = np.sort(e[e>0])[::-1]
#     return 2j / e

# def ak(N):
#     a = []
#     for k in range(2 * N):
#         if k == 0:
#             a += [1/12]
#         else:
#             res = (2 * k + 1)/ (2 * factorial(2 * k + 3))
#             for j,aj in enumerate(a):
#                 res -= aj/factorial(2 * k + 1 - 2*j)
#             a+=[res]
#     return np.array(a)

# def sympy_ak(N):
#     a = []
#     for k in range(2 * N):
#         if k == 0:
#             a += [Rational(1 , 12)]
#         else:
#             res = Rational(2 * k + 1, 2 * factorial(2 * k + 3))
#             for j,aj in enumerate(a):
#                 res -= aj / Rational(factorial(2 * k + 1 - 2*j))
#             a+=[res.simplify()]
#     return a

# def sympy_A(N):
#     aj = sympy_ak(N)
#     def f(i,j):
#         return aj[i-j + N]
#     return Matrix(N, N+1, f)

# def q_vec(N):
#     A = sympy_A(N)
#     q = A.nullspace()
#     assert len(q)
#     q = q[0]
#     q = q / q[0]
#     q.simplify()
#     return q

# def p_q(N):
#     Q  = q_vec(N)
#     aj = sympy_ak(N)
#     p = []
    
#     for i in range(N):
#        a  = aj[0:i+1]
#        a  = a[::-1]
#        q  = Q[0:i+1]
#        res = Rational(0)
#        for j in range(len(a)):
#            res += a[j] * q[j]
#        p+=[res.simplify()]
#     return p, Q

# def sympy_poly(L):
#     from sympy.abc import x
#     l = []
#     for n,v in enumerate(L):
#         l+=[v * x**n]
#     return poly(sum(l))

# def nullspace(A, atol=1e-13, rtol=0):
#     A = np.atleast_2d(A)
#     u, s, vh = svd(A)
#     tol = max(atol, rtol * s[0])
#     nnz = (s >= tol).sum()
#     ns = vh[nnz:].conj().T
#     return ns

# def Amat(N):
#     aj = ak(N)
#     A = np.zeros((N,N+1))
#     for i in range(N):
#         A[i,:] = np.flip(aj[i:i+N+1])
#     return aj, A

# def PSD_coeffs(N):
#     p,q = p_q(N)
#     P   = sympy_poly(p)
#     Q   = sympy_poly(q)
#     dQ  = ddx(Q)
#     poles = PSD_fermi(N).imag
#     print(poles)
#     coeffs = [float(P.eval(-(z**2))/(2 * dQ.eval(-(z**2)))) for z in poles]
#     return poles, coeffs

