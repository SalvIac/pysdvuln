# -*- coding: utf-8 -*-
"""
@author: Salvatore Iacoletti
"""

import numpy as np


def CalResponseSpectra_SP(Tn, dt, acc_g1, damping=0.05):
    '''
    # SDOF Response Spectra
    # Using Piecewise Exact Method and a given vector of frequency points
    # 
    # Jongwon Lee (c)
    # University of Michigan
    # 
    # March 20 2007
    # Jan. 2 2009   Modified for sinusoidal excitation
    # Sept. 1, 2010 Transformed to a function
    # March 17, 2011 PSA and PSV were added.
    # Jan 26, 2022 Salvatore Iacoletti Python version
    
    # [Output]
    # Tnfn_SAVD = a combined matrix consisting of the followings in column:
    #   Tn = natural period of SDOF oscillator (sec)
    #   fn = corresponding natural freqeuncy of SDOF oscillator (Hz)
    #   PSA = (total) pseudo-spectral acceleration in m/s2
    #   SA = (total) spectral acceleration in m/s2
    #   PSV = (relative) pseudo-spectral velocity in m/s.
    #   SV = (relative) spectral velocity in m/s.
    #   SD = (relative) spectral displacement in m.
    
    # [Input]
    # Tn = natural period of SDOF oscillator (sec)
    # dt = time interval. (sec)
    # acc = acceleration time history (m/s2)
    # damping = damping of SDOF system in decimal
    '''
    
    # Generate time history.
    t = np.arange(0., (len(acc_g1)-1e-6) * dt, dt)
    fn = 1/Tn
    wn = 2*np.pi*fn                            # [rad/sec] Natural freq
    beta = damping                             # damping ratio in decimal
    wd = wn*np.sqrt(1 - beta**2)               # damped freq.
    
    # Piecewise Exact Method===================================================
    u_max = np.zeros(len(wd))
    ud_max = np.zeros(len(wd))
    udd_max = np.zeros(len(wd))
    utdd_max = np.zeros(len(wd))
    PSV = np.zeros(len(wd))
    PSA = np.zeros(len(wd))
    for i in range(len(wd)):
        A = np.exp(-beta*wn[i]*dt) * (beta/np.sqrt(1-beta**2) * \
            np.sin(wd[i]*dt)+np.cos(wd[i]*dt))
        B = np.exp(-beta*wn[i]*dt) * (1/wd[i] * np.sin(wd[i]*dt))
        C = 1/wn[i]**2 * (2*beta/wn[i]/dt + np.exp(-beta*wn[i]*dt) * \
            (((1-2*beta**2)/wd[i]/dt - beta/np.sqrt(1-beta**2)) * \
            np.sin(wd[i]*dt) - (1+2*beta/wn[i]/dt) * np.cos(wd[i]*dt)))
        D = 1/wn[i]**2 * (1 - 2*beta/wn[i]/dt + np.exp(-beta*wn[i]*dt) * \
            ((2*beta**2-1)/wd[i]/dt * np.sin(wd[i]*dt) + 2*beta/wn[i]/dt*np.cos(wd[i]*dt)))
    
        A_d = -np.exp(-beta*wn[i]*dt)*(wn[i]/np.sqrt(1-beta**2)*np.sin(wd[i]*dt))
        B_d = np.exp(-beta*wn[i]*dt)*(np.cos(wd[i]*dt) - beta/np.sqrt(1-beta**2)*np.sin(wd[i]*dt))
        C_d = 1/wn[i]**2*(-1/dt + np.exp(-beta*wn[i]*dt) * \
              ((wn[i]/np.sqrt(1-beta**2)+beta/dt/np.sqrt(1-beta**2))*np.sin(wd[i]*dt) + \
              1/dt*np.cos(wd[i]*dt)))
        D_d = 1/wn[i]**2/dt * (1 - np.exp(-beta*wn[i]*dt) * \
              (beta/np.sqrt(1-beta**2)*np.sin(wd[i]*dt) + np.cos(wd[i]*dt)))
    
        u_exact = np.zeros(len(acc_g1))
        ud_exact = np.zeros(len(acc_g1))
        udd_exact = np.zeros(len(acc_g1))
        utdd_exact = np.zeros(len(acc_g1))
        for j in range(len(acc_g1)-1):
            u_exact[j+1] = u_exact[j]*A + ud_exact[j]*B + (-1)*acc_g1[j]*C + (-1)*acc_g1[j+1]*D
            ud_exact[j+1] = u_exact[j]*A_d + ud_exact[j]*B_d + (-1)*acc_g1[j]*C_d + (-1)*acc_g1[j+1]*D_d
            udd_exact[j+1] = -(2*wn[i]*beta*ud_exact[j+1]+wn[i]**2*u_exact[j+1]+acc_g1[j+1])
            utdd_exact[j+1] = udd_exact[j+1] + acc_g1[j+1]
    
        u_max[i] = np.max(np.abs(u_exact))
        ud_max[i] = np.max(np.abs(ud_exact))
        udd_max[i] = np.max(np.abs(udd_exact))
        utdd_max[i] = np.max(np.abs(utdd_exact))
        PSV[i] = u_max[i]*wn[i]
        PSA[i] = PSV[i]*wn[i]
    
    SA = utdd_max   # (Total or Absolute) Spectral Acceleration in m/s2
    SV = ud_max     # (Relative) Spectral Velocity
    SD = u_max      # (Relative) Spectral Displacement
    PSA = PSA       # (Total) Pseudo-spectral Acceleration in g
    PSV = PSV       # (Relative) Pseudo-spectral Velocity
    return dict(Tn=Tn, fn=fn, PSA=PSA, SA=SA, PSV=PSV, SV=SV, SD=SD)
