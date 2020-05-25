import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import pylops
from warping_inversion import warpingInversion, estimateResult
from sinc_interp import sinc_interp_sparse_op, first_diff_sparse_op

plt.close('all')
np.random.seed(10)

def sinc_interp(sig, t, t_new):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html        
    """
    if len(sig) != len(t):
        raise ValueError('sig and t must be the same length')
    # Find the period    
    dt = t[1] - t[0]
    shift_matrix = np.tile(t_new, (len(t), 1)) - np.tile(t[:, np.newaxis], (1, len(t_new)))
    sinc_interp_matrix = np.sinc(shift_matrix / dt)
    y = np.dot(sig, sinc_interp_matrix)
    return y, sinc_interp_matrix

def addAmplitude(mon, dv, wav, alpha):
    return mon - alpha * np.convolve(dv, wav, 'same')

#########################################################################################
###### create input and output

f = 40
sampt = 0.004
halfalpha = 0.02
t = np.arange(0,4,sampt)
refl = np.random.randn(len(t)) / 25
refl[200] = 1
refl[400] = 0.5
refl[600] = -0.3
refl[700] = 0.7
refl[900] = 0.1
rick = pylops.utils.wavelets.ricker(t,f)
rick = rick[0][rick[2]-len(rick[0])//4:rick[2]+len(rick[0])//4+2]
rickD = np.gradient(rick, sampt)
rickDD = np.gradient(rickD, sampt)

dv = np.zeros(len(t))
dv[400:450] = -0.01
dv[700:750] = 0.01
dt = -1 * np.cumsum(dv) * sampt

base = np.convolve(rick, refl, 'same')

mon = estimateResult(base, dt, t, 'dt', sampt, rickD, halfalpha)

plt.plot(t,base)
plt.plot(t,mon)
plt.legend(['base','mon'])
plt.figure()
########## finish creating input and output
#########################################################################


# # Create regularization operator
D2op = pylops.SecondDerivative(len(base), dims=None, dtype='float64')
Dop = pylops.FirstDerivative(len(base), dims=None, dtype='float64')
eye = sparse.spdiags(np.ones(len(t)),0,len(t), len(t))


model_type = 'dv'
regs = [D2op, Dop]
reg_coeffs = [100, 100]


m = warpingInversion(t, base, mon, rick, halfalpha, model_type, Nepochs=61, regs=regs, reg_coeffs=reg_coeffs, m0=None,
 iter_lim=250, show_flag=1, returninfo_flag=0, plot_flag=True, plot_frequency=10)

plt.plot(t, -1*np.cumsum(m)*sampt)
plt.plot(t, dt, color='black', linewidth=2)
plt.show()