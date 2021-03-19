#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:35:11 2021

@author: aleks
"""
import numpy as np
from scipy import sparse as sp
from numba import jit

Inv   = np.linalg.inv
Solve = np.linalg.solve
MM    = np.matmul
Wh    = np.where

def inds_to_lind(e,L):
    ind=None
    ite=0
    for l in L: 
        if e[0]==l[0] and e[1]==l[1]:
            ind=ite
            break
        else:
            ite+=1
    return ind

def error():
    assert 1==0

def get_divisors(n):
    div=[]
    for i in range(1,n//2+1):
        if (n//i-n/i)==0:
            div+=[i]
    return div

def test_partition_2d_sparse_matrix(A,P,tol = 1e-5,sparse=True):
    assert P[-1]==A.shape[0]==A.shape[1]
    assert P[1]-P[0]==P[-1]-P[-2]
    
    if sparse==True:
        i,j,v = sp.find(A)
        ABS=np.abs(v).sum()
        N = len(P)
        Slices_a = [ [slice(P[i],P[i+1]),slice(P[i],P[i+1])] for i in range(N-1)   ]
        Slices_b = [ [slice(P[i],P[i+1]),slice(P[i-1],P[i])] for i in range(1,N-1) ]
        Slices_c = [ [slice(P[i-1],P[i]),slice(P[i],P[i+1])] for i in range(1,N-1) ]
        Abs=0
        for i in range(N-1):
            Abs+=np.abs(A[Slices_a[i][0],Slices_a[i][1]].todense()).sum()
            if i<N-2:
                Abs+=np.abs(A[Slices_b[i][0],Slices_b[i][1]].todense()).sum()
                Abs+=np.abs(A[Slices_c[i][0],Slices_c[i][1]].todense()).sum()
        return Abs/ABS,[Slices_a,Slices_b,Slices_c]

def Transpose(A):
    s=A.shape
    if len(s)==2:
        return A.T
    elif len(s)==3:
        return A.transpose(0,2,1)
    elif len(s)==4:
        return A.transpose(0,1,3,2)
    elif len(s)==5:
        return A.transpose(0,1,2,4,3)


def Build_BTD(A,Slices):
    
    Al = []; Bl = []; Cl = []
    Ia = []; Ib = []; Ic = []
    N=len(Slices[0])
    Sa = Slices[0]
    Sb = Slices[1]
    Sc = Slices[2]
    
    for i in range(N):
        Al+=[np.asarray(A[Sa[i][0],Sa[i][1]].todense())]
        Ia+=[i]
        if i<N-1:
            Bl+=[np.asarray(A[Sb[i][0],Sb[i][1]].todense())]
            Ib+=[i]
            Cl+=[np.asarray(A[Sc[i][0],Sc[i][1]].todense())]
            Ic+=[i]
    
    return Al,Bl,Cl,Ia,Ib,Ic

def Build_BS(A,P):
    vals = []
    inds = []
    I,J,val  = sp.find(A)
    minI = I.min()
    maxI = I.max()
    minJ = J.min()
    maxJ = J.max()
    
    N = len(P)-1
    for i in range(N):
        if P[i]<minI and P[i+1]<minI:
            pass
        elif P[i]>maxI and P[i+1]>maxI:
            pass
        else:
            Si = slice(P[i],P[i+1])
            for j in range(N):
                if P[j]<minJ and P[j+1]<minJ:
                    pass
                elif P[j]>maxJ and P[j+1]>maxJ:
                    pass
                else:
                    Sj = slice(P[j],P[j+1])
                    block = np.asarray(A[Si,Sj].todense())
                    if np.any(block):
                        vals+=[block.copy()]
                        inds+=[[i,j]]
                
    return inds,vals

def Find_copies(L,atol=1e-5,rtol = 1e-5):
    def equal(a,b):
        return np.isclose(a,b,atol = atol , rtol = rtol).all()
    
    n = len(L)
    Truth = np.zeros((n,n),dtype=bool)
    for i in range(n):
        for j in range(i+1,n):
            if equal(L[i],L[j]):
                Truth[i,j] = 1
    


# @jit
# def Check(n,sl,A):
#     T=np.zeros((n,n))
#     for i in range(n):
#         for j in range(n):
#             if i!=j and i!=j+1 and i!=j-1:
#                 T[i,j]=(A[i*sl:(i+1)*sl,j*sl:(j+1)*sl]!=0).any()
#     return T.sum()
# 
# def Is_Matrix_BTD(A,min_div):
#     n1,n2=A.shape[0],A.shape[1]
#     if n1!=n2:
#         print('Matrix needs to be square to have a chance of having an inverse...')
#         return
#     f=get_divisors(n1)
#     fn=[]
#     for i in f:
#         if i>=min_div:
#             fn+=[i]
    
#     print('Found divisors')
#     for n in fn[::-1]:
#         sl=n1//n
#         if Check(n,sl,A)==0:
#             return [sl,n]

def all_zero(A):
    if A is None:
        return True
    else:
        return not np.any(A)

class block_td:
    ##### Articles used: 
    ##### Matthew G Reuter and Judith C Hill
    ##### An efficient, block-by-block algorithm for inverting
    ##### a block tridiagonal, nearly block Toeplitz matrix
    #####
    ##### And
    #####
    ##### Improvements on non-equilibrium and transport 
    ##### Green function techniques: The next-generation transiesta
    
    #Block Tridiagonal  matrix of numpy arrays (vectorised in the first three indecies)#
    def __init__(self,Al,Bl,Cl,I_al,I_bl,I_cl,diagonal_zeros=False,E_grid = None):
        #Matrix-elements and shortcuts to their position in a list
        self.Al=[a.copy() for a in Al]
        self.Bl=[b.copy() for b in Bl]
        self.Cl=[c.copy() for c in Cl]
        self.I_al=I_al.copy()
        self.I_bl=I_bl.copy()
        self.I_cl=I_cl.copy()
        #Sanity checks
        assert len(I_bl)==len(I_cl)
        assert len(I_al)-1==len(I_cl)
        #useful numbers
        self.N=len(I_al)
        self.dt=Al[0].dtype
        self.num_vect_inds = len(Al[0].shape)-2
        #nonzero elements and check for block structure
        self.info(diagonal_zeros)
        self.Shape()
        #initialisations
        self.has_been_conjugated=False
        self.has_been_transposed=False
        self.diagonal_zeros=diagonal_zeros
        self.E_grid = E_grid
    
    def info(self,diagonal_zeros):
        all_slices=[]
        is_zero=[]
        sx = 0
        for i in range(len(self.I_al)):
            shape_i = self.A(i).shape[0+self.num_vect_inds]
            s = []
            z = []
            sy = 0
            for j in range(len(self.I_al)):
                shape_j=self.A(j).shape[1+self.num_vect_inds]
                s += [[slice(sx,sx+shape_i),slice(sy,sy+shape_j)]]
                sy += shape_j
                if i==j or i==j+1 or i+1==j:
                    if diagonal_zeros==False:
                        z+=[1]
                    elif not all_zero(self.Block(i,j)):   #np.count_nonzero(self.Block(i,j))>0:
                        z+=[1]
                    else:
                        z+=[0]
                else:
                    z+=[0]
            all_slices+=[s]
            is_zero+=[z]
            sx+=shape_i
        
        self.all_slices=all_slices
        self.is_zero=np.array(is_zero)
        nonzero_slices=[]
        for i in range(len(self.Al)):
            nZ=[]
            for j in range(len(self.Al)):
                if is_zero[i][j]==1:
                    nZ+=[all_slices[i][j]]
            nonzero_slices+=[nZ]
            
        self.non_zero_slices = nonzero_slices
        self.dtype = self.Al[0].dtype
    
    def _Sequences_XY(self,Print='no'):
        self.Xs=[]; self.Xts=[]
        self.Ys=[]; self.Yts=[]
        x = np.zeros(self.A(self.N-1).shape,dtype=self.dt)
        y = np.zeros(self.A(self.N-1).shape,dtype=self.dt)
        self.Xs+=[x]
        self.Ys+=[x]
        
        for n in range(0,self.N-1):
            yt= Solve(self.A(n)   -y, self.C(n))
            y =MM(self.B(n),yt) #self.B(n). dot(yt)
            self.Yts+=[yt]; self.Ys+= [y]
        
        for n in range(self.N-1,0,-1):
            xt= Solve(self.A(n)-x, self.B(n-1))
            x = MM(self.C(n-1),xt)#self.C(n-1).dot(xt)
            self.Xs +=[x]; self.Xts+=[xt]
        
        
        self.Xs=self.Xs[::-1]
        self.Xts=self.Xts[::-1]
        self.Xts=self.Xts+[None]
        self.Yts=[None]+self.Yts
        if Print=='si':
            print('X,Y,Yt,Xt calculated')
    
    def Gen_Inv_Diag(self):
        self.iMl=[]
        self.inds=[]
        self._Sequences_XY()
        for i in range(self.N):
            self.inds+=[[i,i]]
            self.iMl+=[Inv(self.A(i)-self.X(i)-self.Y(i))]    

    def iM(self,i,j):
        lind=inds_to_lind([i,j],self.inds)
        if lind is not None:
            return self.iMl[lind]
        else: error()
    
    def Invert(self, BW = 'all'):
        print('Calculating matrix elements of inverse\n')
        if BW=='all':
            self.Gen_Inv_Diag()
            #Regner alle elementer i inverse matrix
            for n in range(self.N):
                for m in range(n+1,self.N):
                    self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                    self.inds+=[[m,n]]
                for m in range(n-1,-1,-1):
                    self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                    self.inds+=[[m,n]]
            Res = block_sparse(self.inds.copy(),self.iMl.copy(),self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        elif isinstance(BW,int):
            self.Gen_Inv_Diag()
            #Regner kun elementer et antal steps væk fra diagonalen
            for n in range(self.N):
                for m in range(n+1,min(self.N,n+1+BW)):
                    self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                    self.inds+=[[m,n]]
                for m in range(n-1,max(-1,n-1-BW),-1):
                    self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                    self.inds+=[[m,n]]
            Res = block_sparse(self.inds.copy(),self.iMl.copy(),self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        elif BW[0] == 'N':
            if BW=='N': nn=0 
            else:       nn = int(BW[1:])
            
            self.Gen_Inv_Diag()
            #Regner invers matrix elementer i en N form
            for n in range(self.N):
                if n<=0+nn or n>=self.N-1-nn:
                    for m in range(n+1,self.N):
                        self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                        self.inds+=[[m,n]]
                    for m in range(n-1,-1,-1):
                        self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                        self.inds+=[[m,n]]
            Res = block_sparse(self.inds.copy(),self.iMl.copy(),self.is_zero.shape,E_grid = self.E_grid)
            self.Clean_inverse()
            return Res
        
        elif BW[0]=='Z':
            if BW=='Z': nn=0 
            else:       nn = int(BW[1:])
            
            self.do_transposition()
            self.Gen_Inv_Diag()
            for n in range(self.N):
                if n<=0+nn or n>=self.N-1-nn:
                    for m in range(n+1,self.N):
                        self.iMl +=[MM(-self.Xt(m-1),self.iM(m-1,n))]#[-self.Xt(m-1).dot(self.iM(m-1,n))]
                        self.inds+=[[m,n]]
                    for m in range(n-1,-1,-1):
                        self.iMl +=[MM(-self.Yt(m+1),self.iM(m+1,n))]#[-self.Yt(m+1).dot(self.iM(m+1,n))]
                        self.inds+=[[m,n]]
            self.do_transposition()
            Res = block_sparse(self.inds.copy(),self.iMl.copy(),self.is_zero.shape,E_grid = self.E_grid)
            Res.do_transposition()
            self.Clean_inverse()
            return Res
        
        elif BW[0:3]=='[\]':
            if BW=='[\]':str_nn=''
            else: str_nn=str(int(BW[3:]))
            ResZ = self.Invert(BW='Z'+str_nn)
            ResN = self.Invert(BW='N'+str_nn)
            
            vals_combined = ResZ.vals.copy()
            inds_combined = ResZ.inds.copy()
            is_zero_new = ResZ.is_zero.copy()
            for IJ in ResN.inds:
                if IJ not in ResZ.inds:
                    inds_combined+=[IJ.copy()]
                    vals_combined+=[ResN.Block(IJ[0],IJ[1]).copy()]
                    
            
            Res = block_sparse(inds_combined.copy(),vals_combined.copy(),ResZ.is_zero.shape,E_grid = self.E_grid)
            return Res
        
    def Clean_inverse(self):
        self.inds = None
        self.iMl  = None
        self.inds = []
        self.iMl  = []
        
    def Block(self,i,j):
        if i==j:
            return self.A(i)
        if i==j+1:
            return self.B(j)
        if i==j-1:
            return self.C(i)
        else:
            return None
    
    def A(self,n):
        if n<0 or n>self.N-1:print('Index error A'); error()
        return self.Al[self.I_al[n]]
    def B(self,n):
        if n<0 or n>self.N-2:print('Index error B'); error()
        return self.Bl[self.I_bl[n]]
    def C(self,n):
        if n<0 or n>self.N-2:print('Index error C'); error()
        return self.Cl[self.I_cl[n]]
    def X(self,n):
        if n<0 or n>self.N-1:print('Index error X');error()
        return self.Xs[n]
    def Y(self,n):
        if n<0 or n>self.N-1:print('Index error Y');error()
        return self.Ys[n]
    def Yt(self,n):
        if n<1 or n>self.N-1:print('Index error Yt');error()
        return self.Yts[n]
    def Xt(self,n):
        if n<0 or n>self.N-2:print('Index error Xt');error()
        return self.Xts[n]
    
    def Shape(self):
        n0,n1=0,0
        for i in range(len(self.I_al)):
            n0+=self.A(i).shape[0+self.num_vect_inds]
            n1+=self.A(i).shape[1+self.num_vect_inds]
        
        if self.num_vect_inds==0:
            self.shape=(n0,n1)
        elif self.num_vect_inds==1:
            self.shape=(self.A(i).shape[0],n0,n1)
        elif self.num_vect_inds==2:
            self.shape=(self.A(i).shape[0],self.A(i).shape[1],n0,n1)
        elif self.num_vect_inds==3:
            print('Please visit https://downloadmoreram.com/')
            self.shape=(self.A(i).shape[0],self.A(i).shape[1],self.A(i).shape[2],n0,n1)
        self.Block_shape=(len(self.I_al),len(self.I_al))
    
    def do_transposition(self):
        Na=len(self.Al)
        Nb=len(self.Bl)
        Nc=len(self.Cl)
        Al_new = [Transpose(self.Al[i]).copy() for i in range(Na)] 
        Bl_new = [Transpose(self.Cl[i]).copy() for i in range(Nc)]
        Cl_new = [Transpose(self.Bl[i]).copy() for i in range(Nb)]
        I_al_new = self.I_al.copy()
        I_bl_new = self.I_cl.copy()
        I_cl_new = self.I_bl.copy()
        
        self.Al   = None; self.Bl   = None; self.Cl   = None
        self.I_al = None; self.I_bl = None; self.I_cl = None
        
        self.Al = Al_new
        self.Bl = Bl_new
        self.Cl = Cl_new
        self.I_al = I_al_new.copy()
        self.I_bl = I_bl_new.copy()
        self.I_cl = I_cl_new.copy()
        self.has_been_transposed = not self.has_been_transposed
        self.info(self.diagonal_zeros)
        
    
    def do_conjugation(self):
        Na=len(self.Al)
        Nb=len(self.Bl)
        Nc=len(self.Cl)
        Al_new = [np.conj(self.Al[i]).copy() for i in range(Na)] 
        Bl_new = [np.conj(self.Bl[i]).copy() for i in range(Nc)]
        Cl_new = [np.conj(self.Cl[i]).copy() for i in range(Nb)]
        self.Al   = None
        self.Bl   = None
        self.Cl   = None
        self.Al = Al_new
        self.Bl = Bl_new
        self.Cl = Cl_new
        self.has_been_conjugated = not self.has_been_conjugated
    
    def do_dag(self):
        self.do_transposition()
        self.do_conjugation()
    
    def copy(self):
        new_Al = [self.Al[i].copy() for i in range(len(self.Al))]
        new_Bl = [self.Bl[i].copy() for i in range(len(self.Bl))]
        new_Cl = [self.Cl[i].copy() for i in range(len(self.Cl))]
        if self.E_grid is not None:
            E_grid_new = self.E_grid.copy()
        else:
            E_grid_new = None
        return block_td(new_Al,new_Bl,new_Cl,self.I_al.copy(),self.I_bl.copy(),self.I_cl.copy(),self.diagonal_zeros.copy(),E_grid = E_grid_new)
    
    #Kopieret fra block_sparse
    def Tr(self,Ei = None):
        if Ei is None:
            return block_TRACE(self)
        else:
            return block_TRACE_interpolated(self,Ei)
    
    def TrProd(self,A,Ei1=None,Ei2=None,warning='yes'):
        if Ei1 is None and Ei2 is None:
            return block_TRACEPROD(self,A)
        else:
            return block_TRACEPROD_interpolated(self,A,Ei1,Ei2)
    
    def SumAll(self):
        return block_SUMALL(self)
    
    def SumAllMatrixEntries(self):
        return block_SUMALLMATRIXINDECIES(self)
    
    def BDot(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MATMUL(self,A)
        else:
            return block_MATMUL_interpolated(self,A,Ei1,Ei2)
    
    def Add(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_ADD(self,A)
        else:
            return block_ADD_interpolated(self,A,Ei1,Ei2)
    
    def Subtract(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_SUBTRACT(self,A)
        else:
            return block_SUBTRACT_interpolated(self,A,Ei1,Ei2)
    def MulEleWise(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MULELEMENTWISE(self,A)
        else:
            return block_MULELEMENTWISE_interpolated(self,A,Ei1,Ei2)
    



class block_sparse:
    def __init__(self, inds, vals,Block_shape,E_grid=None,FoRcE_dTypE = None):
        self.inds = inds.copy()
        self.vals = [v.copy() for v in vals]
        self.Block_shape=Block_shape
        self.FoRcE_dTypE = FoRcE_dTypE
        self.info()
        if np.any(self.is_zero):
            self.num_vect_inds = len(vals[0].shape)-2
        else:
            self.num_vec_inds = 0
        self.has_been_conjugated=False
        self.has_been_transposed=False
        self.E_grid = E_grid
        
    
    def Block(self,i,j):
        lind=inds_to_lind([i,j],self.inds)
        if lind is not None:
            return self.vals[lind]
        else: 
            return None
    
    def info(self):
        is_zero=[]
        
        n0=self.Block_shape[0]
        n1=self.Block_shape[1]
        
        for i in range(n0):
            z = []
            for j in range(n1):
                if not all_zero(self.Block(i,j)):
                    z+=[1]
                else:
                    z+=[0]
            
            is_zero+=[z]
        
        self.is_zero=np.array(is_zero)
        if self.FoRcE_dTypE is None:
            self.dtype = self.vals[0].dtype
        else:
            self.dtype = self.FoRcE_dTypE
        
    def do_transposition(self):
        inds_new = [ self.inds[i][::-1].copy()        for i in range(len(self.inds)) ]
        vals_new = [ Transpose(self.vals[i] ).copy()  for i in range(len(self.vals)) ]
        
        self.inds = inds_new.copy()
        self.vals = vals_new.copy()
        self.has_been_transposed = not self.has_been_transposed
        self.Block_shape = self.Block_shape[::-1]
        self.info()
        
    def do_conjugation(self):
        vals_new = [ np.conj(self.vals[i] ).copy()  for i in range(len(self.vals))]
        self.vals = None
        self.vals = vals_new
        self.has_been_conjugated = not self.has_been_conjugated
    def do_dag(self):
        self.do_transposition()
        self.do_conjugation()
    
    def copy(self):
        vals_new = [self.vals[i].copy() for i in range(len(self.vals))]
        inds_new = self.inds.copy()
        if self.E_grid is not None:
            E_grid_new = self.E_grid.copy()
        else:
            E_grid_new = None
        return block_sparse(inds_new, vals_new,self.Block_shape, E_grid = E_grid_new)
    
    def Tr(self,Ei = None):
        if Ei is None:
            return block_TRACE(self)
        else:
            return block_TRACE_interpolated(self,Ei)
    
    def TrProd(self,A,Ei1=None,Ei2=None,warning='yes'):
        if Ei1 is None and Ei2 is None:
            return block_TRACEPROD(self,A)
        else:
            return block_TRACEPROD_interpolated(self,A,Ei1,Ei2)
    
    def SumAll(self):
        return block_SUMALL(self)
    
    def SumAllMatrixEntries(self):
        return block_SUMALLMATRIXINDECIES(self)
    
    def BDot(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MATMUL(self,A)
        else:
            return block_MATMUL_interpolated(self,A,Ei1,Ei2)
    
    def Add(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_ADD(self,A)
        else:
            return block_ADD_interpolated(self,A,Ei1,Ei2)
    def Subtract(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_SUBTRACT(self,A)
        else:
            return block_SUBTRACT_interpolated(self,A,Ei1,Ei2)
    def MulEleWise(self,A,Ei1=None,Ei2=None):
        if Ei1 is None and Ei2 is None:
            return block_MULELEMENTWISE(self,A)
        else:
            return block_MULELEMENTWISE_interpolated(self,A,Ei1,Ei2)


def block_TRACE(A):
    n=A.is_zero.shape[0]
    return sum([np.trace(A.Block(i,i),axis1=-1,axis2=-2) for i in range(n) if A.Block(i,i) is not None])

def block_SUMALLMATRIXINDECIES(A):
    I,J = Wh(A.is_zero>0)
    n_ind = len(I)
    if n_ind==0:
        return 0
    else:
        S=[]
        for tæl in range(n_ind):
            i,j = I[tæl],J[tæl]
            S+=[np.sum(A.Block(i,j),axis=(-1,-2))]
    return sum(S)

def block_SUMALL(A):
    return np.sum(block_SUMALLMATRIXINDECIES(A))

def block_TRACEPROD(A1,A2):
    Res = block_SUMALLMATRIXINDECIES(block_MULELEMENTWISE_TRANSPOSE_LAST(A1, A2))
    return Res

def block_MATMUL(A1,A2):
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    assert A1.Block_shape[1]==A2.Block_shape[0]
    Prod_pat = A1.is_zero.dot(A2.is_zero)
    I,J =  Wh(Prod_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        k1 = Wh(A1.is_zero[i,:]==1)[0]
        k2 = Wh(A2.is_zero[:,j]==1)[0]
        K=np.intersect1d(k1,k2)
        if len(K)>0:
            Res_inds+=[[i,j]]
            First = MM(A1.Block(i,K[0]),A2.Block(K[0],j))
            for k in K[1:]:
                First       += MM(A1.Block(i,k),A2.Block(k,j))
            Res_vals += [First]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A2.is_zero.shape[1]))

def block_ADD(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [A1.Block(i,j)+A2.Block(i,j)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [A1.Block(i,j)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [A2.Block(i,j)              ]
            
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_SUBTRACT(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [ A1.Block(i,j) - A2.Block(i,j)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [ A1.Block(i,j)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [-A2.Block(i,j)              ]
    
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_MULELEMENTWISE(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    Mul_pat = A1.is_zero * A2.is_zero
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[A1.Block(i,j)*A2.Block(i,j)]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)


def block_MULELEMENTWISE_TRANSPOSE_LAST(A1,A2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    Mul_pat = A1.is_zero * A2.is_zero.T
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[A1.Block(i,j)*Transpose(A2.Block(j,i))]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)


def block_TRACE_interpolated(A,Ei):
    assert not isinstance(A.E_grid,type(None))
    F,IND = Interpolate(A.E_grid,Ei)
    n=A.is_zero.shape[0]
    return sum([np.trace(Interpolate_block(A.Block(i,i),F,IND),axis1=-1,axis2=-2) for i in range(n) if A.Block(i,i) is not None])

def block_SUMALLMATRIXINDECIES_interpolated(A,Ei):
    assert not isinstance(A.E_grid,type(None))
    F,IND = Interpolate(A.E_grid,Ei)
    I,J = Wh(A.is_zero>0)
    n_ind = len(I)
    if n_ind==0:
        return 0
    else:
        S=[]
        for tæl in range(n_ind):
            i,j = I[tæl],J[tæl]
            S+=[np.sum(Interpolate_block(A.Block(i,j),F,IND),axis=(-1,-2))]
    return sum(S)

def block_SUMALL_interpolated(A,Ei):
    return np.sum(block_SUMALLMATRIXINDECIES_interpolated(A,Ei))


def block_MULELEMENTWISE_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    #assert A1.num_vect_inds==A2.num_vect_inds
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Mul_pat = A1.is_zero * A2.is_zero
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        
        Res_vals+=[Interpolate_block(A1.Block(i,j),F1,IND1)*Interpolate_block(A2.Block(i,j),F2,IND2)]
        
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)

def block_MULELEMENTWISE_TRANSPOSE_LAST_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Mul_pat = A1.is_zero * A2.is_zero.T
    I,J =  Wh(Mul_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        Res_vals+=[Interpolate_block(A1.Block(i,j),F1,IND1)*Transpose(Interpolate_block(A2.Block(j,i),F2,IND2))]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),A1.is_zero.shape)



def block_TRACEPROD_interpolated(A1,A2,Ei1,Ei2,warning='yes'):
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    ##A1 CANNOT BE THE SAME BTD OR SPARSE BLOCK MATRIX AS A2!!!!!!!!!!!!!
    ##print(A2.has_been_transposed)
    Res = block_SUMALLMATRIXINDECIES(block_MULELEMENTWISE_TRANSPOSE_LAST_interpolated(A1, A2, Ei1, Ei2))
    return Res

def block_MATMUL_interpolated(A1,A2,Ei1,Ei2):
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    assert A1.Block_shape[1]==A2.Block_shape[0]
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1  = Interpolate(A1.E_grid,Ei1)
    F2,IND2  = Interpolate(A2.E_grid,Ei2)
    
    Prod_pat = A1.is_zero.dot(A2.is_zero)
    I,J =  Wh(Prod_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        k1 = Wh(A1.is_zero[i,:]==1)[0]
        k2 = Wh(A2.is_zero[:,j]==1)[0]
        K=np.intersect1d(k1,k2)
        if len(K)>0:
            Res_inds+=[[i,j]]
            First = MM(Interpolate_block(A1.Block(i,K[0]),F1,IND1),Interpolate_block(A2.Block(K[0],j),F2,IND2))
            for k in K[1:]:
                First += MM(Interpolate_block(A1.Block(i,k),F1,IND1),Interpolate_block(A2.Block(k,j),F2,IND2))
            Res_vals += [First]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A2.is_zero.shape[1]))

def block_ADD_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [Interpolate_block(A1.Block(i,j),F1,IND1) + Interpolate_block(A2.Block(i,j),F2,IND2)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [Interpolate_block(A1.Block(i,j),F1,IND1)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [Interpolate_block(A2.Block(i,j),F2,IND2)              ]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

def block_SUBTRACT_interpolated(A1,A2,Ei1,Ei2):
    assert A1.Block_shape==A2.Block_shape
    assert A1.dtype==A2.dtype
    # assert A1.num_vect_inds==A2.num_vect_inds
    
    assert not isinstance(A1.E_grid,type(None))
    assert not isinstance(A2.E_grid,type(None))
    
    F1,IND1 = Interpolate(A1.E_grid,Ei1)
    F2,IND2 = Interpolate(A2.E_grid,Ei2)
    
    Sum_pat = A1.is_zero + A2.is_zero
    I,J =  Wh(Sum_pat>0)
    Res_inds = []
    Res_vals = []
    n_ind =len(I)
    for tæl in range(n_ind):
        i,j = I[tæl],J[tæl]
        Res_inds+=[[i,j]]
        if A1.Block(i,j) is not None and A2.Block(i,j) is not None:
            Res_vals += [ Interpolate_block(A1.Block(i,j),F1,IND1)-Interpolate_block(A2.Block(i,j),F2,IND2)]
        if A1.Block(i,j) is not None and A2.Block(i,j)     is None:
            Res_vals += [ Interpolate_block(A1.Block(i,j),F1,IND1)              ]
        if A1.Block(i,j) is None and A2.Block(i,j)     is not None:
            Res_vals += [-Interpolate_block(A2.Block(i,j),F2,IND2)              ]
    return block_sparse(Res_inds.copy(),Res_vals.copy(),(A1.is_zero.shape[0],A1.is_zero.shape[1]))

@jit
def Interp(E0,E):
    assert E.min()>=E0.min()
    assert E.max()< E0.max()
    F = np.zeros((len(E),2))
    inds = np.zeros((len(E),2))
    ne=len(E)
    ne0=len(E0)
    for j in range(ne):
        e=E[j]
        for i in range(ne0-1):
            if E0[i]<=e<E0[i+1]:
                dE     = E0[i+1]-E0[i]
                de     = e-E0[i]
                f = de/dE
                F[j,0] = 1-f
                F[j,1] = f
                inds[j,0] = i
                inds[j,1] = i+1
                break
    
    return F,inds

def Interpolate(E0,E):
    f,i = Interp(E0,E)
    return f, i.astype(int)

def Interpolate_block(Arr,f,i):
    #Always interpolates in the third index
    if len(Arr.shape)==4:
        return (Arr[:,i[:,0],:,:].transpose(0,2,3,1)*f[:,0]).transpose(0,3,1,2) + (Arr[:,i[:,1],:,:].transpose(0,2,3,1)*f[:,1]).transpose(0,3,1,2)
    elif len(Arr.shape)==2:
        return Arr
    else:
        print('Interpolate_block give array that it isnt compatible with\n')
        assert 1==0
    
def Compare_nd_and_btd(NP, B):
    Bs = B.Block_shape
    T=np.zeros(Bs)
    for i in range(Bs[0]):
        for j in range(Bs[1]):
            
            if len(NP.shape) == 4:
                denseblock = NP[:,:,B.all_slices[i][j][0],B.all_slices[i][j][1]]
            elif len(NP.shape)== 3:
                denseblock = NP[:,B.all_slices[i][j][0],B.all_slices[i][j][1]]
            elif len(NP.shape)== 2:
                denseblock = NP[B.all_slices[i][j][0],B.all_slices[i][j][1]]
            
            if B.Block(i,j) is not None:
                T[i,j]  = np.isclose(denseblock,B.Block(i,j)).all()
            elif B.Block(i,j) is None and (denseblock==0).all():
                T[i,j] = 1
    return T




