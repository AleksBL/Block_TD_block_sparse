import numpy as np
from numpy.linalg import svd
from numba import njit, prange
from scipy.signal import hilbert
import os
import sys
file_p = (__file__[:-8])
sys.path.append(file_p)

from Croyk0nfig import CACHE, PARALLEL



@njit(cache = CACHE)
def L(E, W, ei):
    return W**2 /( (E - ei)**2  + W**2 )

@njit(cache = CACHE)
def Lvec(E, W, ei):
    return W**2 /( (E - ei)**2  + W**2 )

@njit(cache = CACHE)
def Overlap(ei, wi, ej, wj):
    xi = ei + 1j*wi
    xj = ej + 1j*wj
    if abs(xi -  xj)< 1e-12:
        res = np.pi * wi/2
    else:
        res = 2j * np.pi/4 * wi * wj *( 1/(xj-np.conj(xi)) + 1/(xi - np.conj(xj)) - 1/(xi - xj) - 1/(xj - xi))
    return res.real

@njit(cache = CACHE)
def d_Overlap_d_eps(ei, wi, ej, wj):
    eps = 1e-5
    return (Overlap(ei + eps, wi, ej, wj) - Overlap(ei - eps, wi, ej, wj)) /(2 * eps)

@njit(cache = CACHE)
def d_Overlap_d_gamma(ei, wi, ej, wj):
    eps = 1e-5
    return (Overlap(ei, wi + eps, ej, wj) - Overlap(ei, wi - eps, ej, wj)) /(2 * eps)

@njit(cache = CACHE)
def d_Overlap_d_eps_same(ei,wi):
    eps = 1e-5
    return (Overlap(ei + eps, wi, ei + eps, wi) - Overlap(ei - eps, wi, ei - eps, wi)) /(2 * eps)
@njit(cache = CACHE)
def d_Overlap_d_gamma_same(ei,wi):
    eps = 1e-5
    return (Overlap(ei, wi + eps, ei, wi + eps) - Overlap(ei, wi - eps, ei, wi - eps)) /(2 * eps)

atan = np.arctan
ln = np.log
@njit(cache = CACHE)
def hat_with_lrtz(x1,x2,x3, ej, gj):
    t1 = (ej - x1)/(x2-x1)  * (-gj) * (atan((ej-x2)/gj) - atan((ej-x1)/gj))
    t2 = gj**2/(2 *(x2-x1)) * (ln((x2 - ej)**2 + gj**2) - ln((x1 - ej)**2 + gj**2))
    t3 = (ej - x3)/(x3-x2)  * (-gj) * (atan((ej-x3)/gj) - atan((ej-x2)/gj))
    t4 = gj**2/(2 *(x3-x2)) * (ln((x3 - ej)**2 + gj**2) - ln((x2 - ej)**2 + gj**2))
    res = t1+t2-t3-t4
    return res.real

@njit(cache = CACHE)
def d_hat_with_lrtz_d_eps(x1,x2,x3,ej,gj):
    eps = 1e-5
    return (hat_with_lrtz(x1,x2,x3, ej + eps, gj) - hat_with_lrtz(x1,x2,x3, ej - eps, gj)) /(2 * eps)

@njit(cache = CACHE)
def d_hat_with_lrtz_d_gamma(x1,x2,x3,ej,gj):
    eps = 1e-5
    return (hat_with_lrtz(x1,x2,x3, ej, gj + eps) - hat_with_lrtz(x1,x2,x3, ej, gj - eps)) /(2 * eps)

@njit(cache = CACHE)
def delta_error_linear_lorentzian(xi, fi, PARS):
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
    
    eps  = 1.0 # make it agree with jacobian of error
    res2 = 0.0
    
    x1,x2,x3 = xi[0]-eps, xi[0], xi[1]
    for j in range(nl):
        res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[0] * GI[j]
    
    x1,x2,x3 = (xi[ns-2], xi[ns-1], xi[ns-1] + eps)
    for j in range(nl):
        res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[ns-1] * GI[j]
    
    for i in range(1,ns-1):
        x1,x2,x3 = xi[i-1:i+2]
        for j in range(nl):
            res2 += hat_with_lrtz(x1, x2, x3, EI[j], WI[j]) * fi[i] * GI[j]
    
    return res - 2 * res2

@njit(cache = CACHE, parallel = PARALLEL)
def delta_error_many_linear_lorentzian(xi, fiv, PARS_v):
    # xiv : ( n), sampling points
    # fiv : (ne, n), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    for i in prange(n):
        res += delta_error_linear_lorentzian(xi, fiv[:,i], PARS_v[:, :, i]).real
    return res

@njit(cache = CACHE,parallel = PARALLEL)
def delta_error_many_linear_lorentzian_complex(xi, fiv, PARS_v):
    # xiv : ( n), sampling points
    # fiv : (ne, n), values
    # Pars_v: (3, NL, nb)
    n = len(fiv[0,:])
    res = 0.0
    PARS_vi = np.zeros(PARS_v.shape)
    PARS_vi[0:2,:,:] = PARS_v[0:2, :,:].real
    PARS_vi[2,  :,:] = PARS_v[2  , :,:].imag
    
    for i in prange(n):
        res += delta_error_linear_lorentzian(xi, fiv[:,i].real, PARS_v [:, :, i].real).real + \
               delta_error_linear_lorentzian(xi, fiv[:,i].imag, PARS_vi[:, :, i].real).real
    
    return res

@njit(cache = CACHE)
def lrz_overlap_matrix(e,w):
    # O is normalised overlap
    # o is the numbers used to normalise
    n = len(e)
    o = np.zeros(n); O = np.zeros((n,n))
    for i in range(n):
        o[i] = Overlap(e[i],w[i],e[i],w[i])
    for i in range(n):
        for j in range(n):
            if i!=j:
                O[i,j] = Overlap(e[i],w[i],e[j],w[j] ) / np.sqrt(o[i] * o[j])
    return O,o

@njit(cache = CACHE)
def Punish_overlap(e,w):
    O,o = lrz_overlap_matrix(e,w)
    return np.tan(O * np.pi/2).sum()

@njit(cache =CACHE)
def d_normed_overlap_d_gamma(ei,wi,ej,wj):
    eps = 1e-5
    Oij_1 = Overlap(ei,wi + eps, ej, wj)
    Oij_2 = Overlap(ei,wi - eps, ej, wj)
    Oii_1 = Overlap(ei,wi+eps,ei,wi+eps)
    Oii_2 = Overlap(ei,wi-eps,ei,wi-eps)
    Ojj   = Overlap(ej,wj,ej,wj)
    return (Oij_1/np.sqrt(Oii_1*Ojj) - Oij_2/np.sqrt(Oii_2*Ojj))/(2*eps)

@njit(cache = CACHE)
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

@njit(cache = CACHE)
def grad_Punish_overlap(e,w):
    n = len(e)
    grad = np.zeros((2, n))
    O,o  = lrz_overlap_matrix(e, w)
    os   = np.sqrt(o)
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

@njit(cache = CACHE)
def d_tan_dx(x):
    return np.pi/2*(1/np.cos(x*np.pi/2))**2



@njit(cache = CACHE, parallel = PARALLEL)
def jacobian_delta_error_linear_lorentzian(xi, fiv, ei, wi, Gi_v):
    nb   = len(fiv[0,:])
    res  = np.zeros(len(ei) + len(wi) + len(Gi_v[:,0]) * len(Gi_v[0,:]))
    n_g  = len(ei)
    nfi  = len(xi)
    it   = 0
    eps  = 1.0 #make it agree with total error
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
    
        # for b in range(nb):
        #     for m in range(n_g):
        #         res[idx3[i,b]] += 2 * Gi_v[m,b] * Oij[i,m]#Overlap(ei[i],wi[i],ei[m],wi[m])
            
        #     for ii in range(nfi):
        #         res[idx3[i,b]] -= 2 * fiv[ii,b] * hwl[ii, i]#hat_with_lrtz(x1,x2,x3, ei[i], wi[i])
    
    return res

@njit(cache = CACHE)
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

def ez_wrap(xi, fiv, ei, wi, Gi_v):
    PARS = np.zeros((3,)+ei.shape+fiv[0,:].shape, dtype = np.complex128)
    PARS[0,:,:] = wi[:, np.newaxis]
    PARS[1,:,:] = ei[:, np.newaxis]
    PARS[2,:,:] = Gi_v
    return delta_error_many_linear_lorentzian_complex(xi, fiv, PARS)



# def L1L2(x):
#     return L(x , 0.01, -0.0) * L(x , 0.01001, 0)

@njit(cache = CACHE)
def L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0,:])
    res = np.zeros(len(E), dtype = np.complex128)
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l])
    return res


@njit(cache = CACHE)
def Lorentzian_basis(f, zi, ei, gamma):
    assert len(zi) == len(ei) == len(gamma)
    nl =  len(zi)
    B  =  np.zeros((nl,nl), dtype = np.complex128)
    for i in range(nl):
        for j in range(nl):
            B[i,j] = L(zi[i], gamma[j], ei[j])
    return np.linalg.inv(B).dot(f)

@njit(cache = CACHE)
def Matrix_lorentzian_basis(M, zi, ei, gamma):
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

@njit(cache = CACHE)
def evaluate_Lorentz_basis_matrix(M, E, ei, gamma):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni, nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                res[ik,:,i,j] = L_sum(E, pars)
    
    return res

@njit(cache = CACHE)
def evaluate_KK_matrix(M, E, ei, gamma):
    nk = M.shape[0]
    ne = M.shape[1]
    ni = M.shape[2]
    nj = M.shape[3]
    res = np.zeros((nk, len(E), ni,nj), dtype = np.complex128)
    for ik in range(nk):
        for i in range(ni):
            for j in range(nj):
                pars = np.vstack((gamma[ik], ei[ik], M[ik,:,i,j]))
                res[ik,:,i,j] = KK_L_sum(E, pars)
    return res

@njit(cache = CACHE)
def KK_L_sum(E, var):
    Wi = var[0]
    ei = var[1]
    Gi = var[2]
    nl = len(var[0,:])
    res = np.zeros(len(E), dtype = np.complex128)
    for l in range(nl):
        res += Gi[l] * Lvec(E, Wi[l], ei[l]) * (E - ei[l]) / Wi[l]
    return res




offset_E = 1.0
@njit(cache = CACHE)
def gamma_from_centres_matrix(ei,wi):
    n = len(ei)
    Mat = np.zeros((n,n),dtype = np.complex128)
    for i in range(n):
        for j in range(n):
            Mat[i,j] = Overlap(ei[i], wi[i], ei[j], wi[j])
    return Mat

@njit(cache = CACHE)
def gamma_from_centers_RHS(ei, wi, fi, xi):
    n = len(ei)
    nf = len(fi)
    v   = np.zeros(n, dtype = np.complex128)
    for i in range(n):
        for j in range(nf):
            if j == 0:
                x1,x2,x3 = xi[j]-offset_E,xi[j],xi[j+1]
            elif j == nf-1:
                x1,x2,x3 = xi[j-1],xi[j],xi[j]+offset_E
            else:
                x1,x2,x3 = xi[j-1],xi[j],xi[j+1]
            
            v[i]+=fi[j] * hat_with_lrtz (x1,x2,x3, ei[i], wi[i])
    return v

#@njit(cache = True)
def gammas_from_centers_vec(ei,wi,fiv,xi):
    nl = len(ei)
    nb = len(fiv[0,:])
    Gi = np.zeros((nl,nb), dtype = np.complex128)
    Mat = gamma_from_centres_matrix(ei,wi)
    
    try:
        iM=np.linalg.inv(Mat)
        for b in range(nb):
            Gi[:,b] = iM.dot(gamma_from_centers_RHS(ei,wi,fiv[:,b], xi))
    except:
        print(ei, wi)
        
        Mat = gamma_from_centres_matrix(ei + np.random.random(nl)*0.01,wi)
        iM = np.linalg.inv(Mat)
        for b in range(nb):
            Gi[:,b] = np.dot(iM, gamma_from_centers_RHS(ei,wi,fiv[:,b], xi))
    return Gi




# xi = np.linspace(-2,2,100)
# ei = np.linspace(-1.5, 1.5, 20)
# wi = np.random.random(20)
# Gi_v = np.random.random((20, 6))+1j * np.random.random((20, 6))
# fiv = np.array([np.sin(xi) + 1j * np.cos(xi)]*6).T
# de = np.zeros(20); de[2] = 1e-4
# dg = np.zeros(Gi_v.shape,dtype = np.complex128); dg[0, 2] =  1j*1e-4
# g1 = (ez_wrap(xi, fiv, ei+de, wi, Gi_v) - ez_wrap(xi, fiv, ei-de, wi, Gi_v)) / 2e-4
# h1 = (ez_wrap(xi, fiv, ei, wi, Gi_v + dg) - ez_wrap(xi, fiv, ei, wi, Gi_v - dg)) / 2e-4
# g2 = jacobian_delta_error_linear_lorentzian_complex(xi, fiv, ei, wi, Gi_v)
# Gi_v2 = gammas_from_centers_vec(ei,wi,fiv,xi)

# import matplotlib.pyplot as plt



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

