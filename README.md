# Block_TD_block_sparse

Matrix classes for matrices that are block-tridiagonal and zero elsewhere, and simply "block sparse" - meaning they only have a few non-zero blocks, for which the data is stored in numpy arrays here. We are somewhere between the sparse scipy-class, but with some of the speed that comes with smaller dense arrays. These types of matrices are often seen in models for electronic structure in a localised orbital basis in quantum transport calculations.

Contains inversion algorithm for a block-tridiagonal matrix as the the one found in found in e.g. "Matthew G Reuter and Judith C Hill An efficient, 
block-by-block algorithm for inverting a block tridiagonal, nearly block Toeplitz matrix" and "Improvements on non-equilibrium and transport Green function techniques: The next-generation TRANSIESTA" by: Nick Rübner Papior, Nicolás Lorente, Thomas Frederiksen, Alberto García, Mads Brandbyge (sorry whomever came up with the algorithm, havent found the original paper).  

Inversion is a pesky O(2.4) operation, meaning it is favorable use the block-tridiagonal structure of a matrix if it happens to be structured in such a way.

Relies on numpy broadcasting meaning the block matrix can be of (K,L,N,i,j), where i,j are the actual matrix indecies. 

You should just the the "Block_matrices.py"-script and put it in a folder where you want to do your calculations
The basic workings are as described below, but can also do a bit more advanced stuff, like linear elementwise interpolation in the "N"-index, make matrices from the scipy sparse format, and [sisl](http://zerothi.github.io/sisl/docs/latest/index.html) Hamiltonians.

The python package "sparse" will also be implemented at some point. 

Also, see [siesta_python](https://github.com/AleksBL/siesta_python) for how to get the greens function for an electronic structure in this format

## Write to abalo@dtu.dk if you want something implemented

## How to do calculations
## (Start of python-code)
```
import numpy as np
from Block_matrices import block_sparse, block_td
import matplotlib.pyplot as plt
r = np.random.random
```


## Matrix  A : 
```
block_shape = (3,3)
vals = [r((3,5,5)), r((3,5,5)),r((3,5,5))]
inds = [[0,0],[1,2],[2,2]]
A = block_sparse(inds,vals,block_shape)
```

## Matrix B: 
```
block_shape = (3,3)
vals = [r((3,5,5)),r((3,5,5))]
inds = [[0,0],[2,0]]
B = block_sparse(inds,vals,block_shape)
```

#### We can do matrix multiplication with the, where only the 
#### two last indecies count as matrix indecies, the rest uses numpy's broadcasting
```
C = A.BDot(B)
```
#### We can copy arrays
```
C2 = C.copy()
#but only tranpose them manually, which modifies all the values
C2.do_transposition()
D = A.BDot(C2)


for i in range(2):
    D = B.BDot(D)
    D.do_transposition()
```

#### We can to a trace
```
print(D.Tr())
```


#### We can do a trace with a product of itself (and of course any other matrix with the same block-shape)
```
print(D.TrProd(D))
```

the block_sparse class can only do matrix-multiplication and traces (and maybe more more if anybody asks). 
Matrices should be have a lot of zeros for this to be efficient, as the python-for loops will become apparent if there are 
many nonzero entries

## The "Block-tridiagonal" part

### Lets make a "A0-A1-A1-A2-A3" Block-tridiagonal matrix
```
s1 = 2
s2 = 4
a0 = r((3,s1,s1))
a1 = r((3,s2,s2))
a2 = r((3,s1,s1))
a3 = r((3,s1,s1))

Al = [a0, a1, a2, a3];                                      Ia = [0,1,1,2,3] # Notice we can place the same block several places
Bl = [r((3,s2,s1)),r((3,s2,s2)),r((3,s1,s2)),r((3,s1,s1))]; Ib = [0,1,2,3]
Cl = [r((3,s1,s2)),r((3,s2,s2)),r((3,s2,s1)),r((3,s1,s1))]; Ic = [0,1,2,3]


BTD = block_td(Al,Bl,Cl,Ia,Ib,Ic)
iBTD = BTD.Invert()
```


### Taking the trace of the their product, we should hopefully be taking the trace of the identity-matrix
###which should yield the dimension of the matrix: 

```
print(BTD.TrProd(iBTD))
```

### The individual blocks of the matrix is accessed as
```
P = BTD.Block(0,1)
plt.matshow(P[0,:,:])
```

### We can multiply the BTD matrix with a block-sparse matrix: 
```
K = block_sparse([[0,0]],[2*a0],(5,5))
print(K.BDot(iBTD).Tr())
```

### Manually inverting the equivalent matrix with numpy gives:
```
Full = np.zeros(BTD.shape)
S = [0,2,6,10,12,14]
for i in range(5):
    Full[:,S[i]:S[i+1],S[i]:S[i+1]]  = Al [Ia[i]]
    if i<5-1:
        Full [:,S[i]:S[i+1],S[i+1]:S[i+2]]  = Cl [Ic[i]]
        Full [:,S[i+1]:S[i+2],S[i]:S[i+1]]  = Bl [Ib[i]]
iFull = np.linalg.inv(Full)

i,j  = 0,2

print(np.isclose(iBTD.Block(i,j),iFull[:,S[i]:S[i+1],S[j]:S[j+1]]).all())



```
### This algorthm is does of course also work on small matrices, but the advatantages over normal inversion of dense matrices is evident when considering e.g a 2000 x 2000 matrix and we restrict ourselves to e.g. the diagonal blocks:
### This script should take about a minute to run and might convince you to use this code if whatever problem you want to solve involves block-tridiagonal matrices:

```
from Block_matrices import block_td
import numpy  as np
from time import time

inv = np.linalg.inv
r = np.random.random
D1     = 2
D2     = 2
N_test = 20


S      = [np.random.randint(100,120) for i in range(N_test)]
Ia=[i for i in range(N_test   )]
Ib=[i for i in range(N_test-1 )]
Ic=[i for i in range(N_test-1 )]

Al = [r((D1,D2,S[i  ],S[i]))+1j*r((D1,D2,S[i  ],S[i])) for i in range(N_test)]
Bl = [r((D1,D2,S[i+1],S[i]))+1j*r((D1,D2,S[i+1],S[i])) for i in range(N_test-1)]
Cl = [r((D1,D2,S[i],S[i+1]))+1j*r((D1,D2,S[i],S[i+1])) for i in range(N_test-1)]

Full  = np.zeros((D1,D2,sum(S),sum(S)),dtype=np.complex128)
Slices=[]
SUM=0

for i in range(N_test):
    Slices+=[slice(SUM,SUM+S[i])]
    SUM+=S[i]

for i in range(N_test):
    Full[:,:,Slices[i],Slices[i]]  = Al [i]
    if i<N_test-1:
        Full [:,:,Slices[i],Slices[i+1]]  =  Cl [i]
        Full [:,:,Slices[i+1],Slices[i]]  =  Bl [i]
        

BTD  = block_td(Al,Bl,Cl,Ia,Ib,Ic, 
                )
s1 = time()
for i in range(5):
    iFull = inv(Full)

s2 = time()
for i in range(5):
    iBTD = BTD.Invert( BW= 0)# BW = 0 means the diagonal, and can take several different keywords, look at the function in Block_matrices.py
s3 = time()
print('five dense inversions took : ',  s2-s1, ' seconds')
print('five BTD inversions took : ',  s3-s2, ' seconds')

#check which block you want (of the diagonals at least, but more if you change BW to something else):
i,j  = 2,2
si, sj =  BTD.all_slices[i][j]

print('elements close: ',np.isclose(iFull[..., si,sj], iBTD.Block(i,j)).all() )

```
A handy function for testing is the "Blocksparse2Numpy" function, which takes a block_sparse or block_td class and returns the equivalent dense numpy array:

```
from Block_matrices import Blocksparse2Numpy
iBTD = BTD.Invert(); # find all blocks of inverse
BTD_dense = Blocksparse2Numpy(BTD, BTD.all_slices)
print('BTD converted to dense is close to Full: ', np.isclose(BTD_dense, Full).all())

iBTD_dense = Blocksparse2Numpy(iBTD, BTD.all_slices) # note the iBTD does not have the "all_slices" attribute, but it shares it will BTD
print('BTD converted to dense is close to Full: ', np.isclose(iBTD_dense, iFull).all())



