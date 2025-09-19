CACHE    = False     # numba
PARALLEL = True      # numba
FASTMATH = True      # numba
dE_fitting = .5      # ramp outside of fitting window
L_cutoff   = 2.5e-5  # cutoff for tails of Lorentzian basis functions
direct_eig_n     = 5 # for direct eigenvalue PSD enforcement
direct_eig_nW    = 5 
direct_eig_lbtol =-1e-2
direct_eig_maxit = 5000
direct_eig_fact  = 0.01
direct_eig_eps   = 0.1
Gamma_fallback   = 'pseudo-inverse' # "random" or "pseudo-inverse"
Gamma_default    = 'Inversion' # 'Inversion' or 'Curvefit'
if False:
    print('Fitting module initialised!')
