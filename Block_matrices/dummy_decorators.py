# These decorators FAKE numba njit, jit and prange
# They do absolutely nothing other than making people without numba able to run the code
# FAKE
def njit(cache=False, parallel=False):
    def dec(f):
        return f
    return dec
#FAKE
def jit(cache=False, parallel=False):
    def dec(f):
        return f
    return dec
prange=range
