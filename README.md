# Block_TD_block_sparse

Matrix classes for matrices that are block-tridiagonal and zero elsewhere, and simply "block sparse" - meaning they only have a few non-zero blocks, for which the data is stored in numpy arrays here. We are somewhere between the sparse scipy-class, but with some of the speed that comes with smaller dense arrays.  

Contains inversion algorithm for a block-tridiagonal matrix as the the one found in found in e.g. "Matthew G Reuter and Judith C Hill An efficient, 
block-by-block algorithm for inverting a block tridiagonal, nearly block Toeplitz matrix" (the simple, nontoeplitz one, I havent looked at the original paper, 
sorry whomever came up with the algorithm) .  

Relies on numpy broadcasting meaning the block matrix can be of (K,L,N,i,j), where i,j are the actual matrix indecies. 

You should just the the "Block_matrices.py"-script and put it in a folder where you want to do your calculations
The basic workings are as described below, but can also do a bit more advanced stuff, like interpolation in the second index. 
It can have "arrays" with up to five indecies, where the two last are the matrix like ones, and the rest works like numpy broadcasting

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


###The block_sparse class can only really do matrix-multiplication and traces. 
###Matrices should be have a lot of zeros for this to be efficient, as the for python-for loops will become apparent if there are 
###many nonzero entries

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
