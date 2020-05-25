import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse

import pylops

def sinc_interp(sig, t, t_new):
    """
    Interpolates x, sampled at "s" instants
    Output y is sampled at "u" instants ("u" for "upsampled")
    
    from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html  

    from: https://gist.github.com/endolith/1297227      
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
    wav_small = wav[np.abs(wav)>1e-20]
    nismall = len(wav_small)
    # estimate half length of wavelet - edge points within this interval will show edge effects from
    # convolution and need to be set to zero.
    out = alpha * np.convolve(dv, wav, 'same')
    out[:nismall//2] = 0
    out[-nismall//2:] = 0
    return mon - out

def convMat(input):
    '''
    Generate convolutional matrix
    '''
    ni = len(input)
    input = input[np.abs(input)>1e-20]
    nismall = len(input)
    diag_data = [i*np.ones(ni) for i in input]
    diags = range(-nismall//2+1,nismall//2+1)
    mat = sparse.spdiags(diag_data,diags,ni,ni)
    # can't expand and contract dimensions of probelm here so we'll have to blank out the convolution
    # that would only invovle part of the wavelet at the start and end of the trace. 
    # Alternative is to expand the trace in the initial problem and solve for a large model dimension then 
    # zero out the edges of the model after each iteration. This also has problems and makes dimensionality 
    # of the problem larger.
    mat = sparse.lil_matrix(mat)
    mat[0:nismall//2,:] = 0
    mat[ni-nismall//2:,:] = 0
    return mat

def makeForwardOperator(mon_hat, wav, sampt, halfalpha, model_type, m, count, no_amp_term_pass):
    '''
    Function to make Jacobian for seismic trace warping
    '''
    N = len(mon_hat)
    if model_type == 'dv':
        wavD = np.gradient(wav, sampt)
        ampJ = convMat(wavD)
        forward_op = sparse.spdiags([-1 * np.gradient(mon_hat, sampt)],[0], N, N)
        if not no_amp_term_pass:
            forward_op -= halfalpha * ampJ
    else:
        wavDD = np.gradient(np.gradient(wav, sampt), sampt)
        ampJ = convMat(wavDD)
        forward_op = sparse.spdiags([np.gradient(mon_hat, sampt)],[0], N, N)
        if not no_amp_term_pass:
            forward_op -= halfalpha * ampJ
    return forward_op

def estimateResult(base, m, t, model_type, sampt, waveletD, halfalpha):
    '''
    Function to estimate warped base trace
    '''
    if model_type=='dv':
        m2 = -1 * np.cumsum(m) * sampt
    else:   
        m2 = m
    mon_hat,_ = sinc_interp(base, t, t - m2)
    mon_hat_orig = mon_hat
    if model_type=='dt':
        m2 = -1 * np.gradient(m, sampt)
    else:   
        m2 = m
    mon_hat = addAmplitude(mon_hat, m2, waveletD, halfalpha)
    return mon_hat


def warpingInversion(t, base, mon, wavelet, halfalpha, model_type, Nepochs=50, regs=[], reg_coeffs=[], m0=None,
 iter_lim=250, show_flag=1, returninfo_flag=0, plot_flag=False, plot_frequency=10, no_amp_term_pass=False):
    '''
    Main inersion function. Takes all arguments, checks some (to be completed), runs non-linear inversion.
    TODO - Add more solvers
         - Better documentation
         - Add support for multiple traces
         - Generalize this function
    '''

    # check inputs
    if model_type is not 'dv' and model_type is not 'dt':
        raise ValueError("model_type must be either 'dv' or 'dt'")
    if m0 is not None:
        if len(m0)!=len(base):
            raise ValueError("length of initial model guess must be equal to the length of the base seismic")

    # check initial guess of model or set to zeros
    if m0 is None:
        m = np.zeros(len(base))
    else:
        m = m0
    # define sample interval
    sampt = t[1]-t[0]

    # define wavelet first derivative
    waveletD = np.gradient(wavelet, sampt)

    leg = ['']
    # Gauss-Newton inversion
    for count in range(Nepochs):

        # estimate result
        mon_hat = estimateResult(base, m, t, model_type, sampt, waveletD, halfalpha)
        # make jacobian
        forward_op = makeForwardOperator(mon_hat, wavelet, sampt, halfalpha, model_type, m, count, no_amp_term_pass)
        # form residual
        res = mon-mon_hat
        # run inversion
        xreg = \
        pylops.optimization.leastsquares.RegularizedInversion(forward_op, regs, res,
                                                            epsRs=reg_coeffs,
                                                            returninfo=returninfo_flag,
                                                            **dict(iter_lim=iter_lim,
                                                                    show=show_flag))
        # update model
        m -= xreg
        # plot if necessary
        if plot_flag and count % plot_frequency == 0:
            plt.plot(t,m)
            leg.append(f'Iteration: {count}')
    if plot_flag:
        plt.legend(leg)
        plt.show()
    return m