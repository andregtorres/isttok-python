#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 14:21:42 2018

@author: bernardo
From
Model-Based Approach for Magnetic
Reconstruction in Axisymmetric
Nuclear Fusion Machines by Cenedese et al
"""

from __future__ import print_function

import numpy as np
#    np.set_printoptions(precision=3)
from scipy import signal
import matplotlib.pyplot as plt
#import scipy as scp
#import scipy.constants as cnst

import magnetic_flux_fields as mf
# ISTTOK Geometric parameters
from isttok_magnetics import isttok_mag, isttok_mag_1, isttok_mag_2
from getSdasSignal import getSignal
from getMirnov import FsamplingMARTe, ch_prim, getMirnovInt, plotAllPoloid, plotAllPoloid2
from StartSdas import StartSdas

def buildLtiModelB(RIc, ZIc, isttok_mag):
    """
    Build LTI system state model for a asissymetric Tokamak with passive conductors
    # State variable: Psi_c (Pol Flux at the eddy current positions)
    # SS Model: dot(Psi_c) = A * Psi_c + [Bpl  Bs] [ip, is]
    #                   ic = C * Psi_c + [Dpl  Ds] [ipl, is]
    # Dpl, DPl not yet calculates
    #
    #https://apmonitor.com/pdc/index.php/Main/ModelSimulation

    Args:
        nc: number of passive copper shell filaments
        ns: number of PFC active coils (sources)
        RIc, ZIc: arrays with position of passive conductors

    Returns:
    """

    ResCopper = 1.0e-4 # 0.1 mOhm
    aCopper = 10.0e-3  # 'wire' radius 10 mm
    ns = 3 # number of active coil circuits
    nc= len(RIc)

    # mutual inductance matrices between the elements of the passive structure
    Mcc=np.zeros((nc,nc))

    # mutual inductance matrices of the elements of the passive structure
    # and active coils
    Mcs=np.zeros((nc,ns))
    # diagonal Resistance matrix Rc
    Rc = np.diag([1.0]*nc) * ResCopper

    #RIc = ISTTOK['Rcopper'] * np.ones(nc) # Major radius of shell 'wires'
    for i in range(nc):
        Mcc[i,i] = mf.selfLoop(RIc[i], aCopper)
        for j in range(i+1, nc):
            Mcc[i,j] = mf.mutualL(RIc[i], ZIc[i],RIc[j], ZIc[j])
            Mcc[j,i] = Mcc[i,j]

    for i in range(ns):
        for k in range(len(isttok_mag['TurnsPfcVer'])):
            Mcs[i,0] += isttok_mag['TurnsPfcVer'][k] * mf.mutualL(isttok_mag['RPfcVer'][k], isttok_mag['ZPfcVer'][k], RIc[i], ZIc[i])
        for k in range(len(isttok_mag['TurnsPfcHor'])):
            Mcs[i,1] += isttok_mag['TurnsPfcVer'][k] * mf.mutualL(isttok_mag['RPfcHor'][k], isttok_mag['ZPfcHor'][k], RIc[i], ZIc[i])
        for k in range(len(isttok_mag['TurnsPfcPrim'])):
            Mcs[i,2] += isttok_mag['TurnsPfcPrim'][k] * mf.mutualL(isttok_mag['RPfcPrim'][k], isttok_mag['RPfcPrim'][k], RIc[i], ZIc[i])

    # Model dot(Psi_c) = A * Psi_c + [Bp + Bpfc] [ip; ipfc]
    # ic = C * Psi_c + [Dp + Dpfc] [ip; ipfc]
    # State variable Psi_c (Pol Flux at the eddy current positions)
    #
    invMcc = np.linalg.inv(Mcc)
    A = -np.matmul(Rc,invMcc)
    Bs = -np.matmul(A,Mcs)
    C = invMcc
    Ds =-np.matmul(invMcc,Mcs)

    return A, Bs, C, Ds


def buildIs2BpolB(isttok_mag):
    """
    Build B poloidal response Matrix on the poloidal field probes from a set of
    PFC coil circuits (Vertical + Horizontal)
    Gives poloidal field on each probe for a Is=1A on coils

    Args:

    Returns:
        Is2Bpol :
    """
    ns = 3 # number of PFC active independent coils circuits (sources)
    #       number of poloidal probes
    nPrb = isttok_mag['nPrb']

    Rprb = isttok_mag['Rprb']
    Zprb = isttok_mag['Zprb']
    tethaProb = isttok_mag['tethaPrb']

    turnsV  = isttok_mag['TurnsPfcVer']
    RPfcVer = isttok_mag['RPfcVer']
    ZPfcVer = isttok_mag['ZPfcVer']
    IgainVert = isttok_mag['IgainVert']

    turnsH  = isttok_mag['TurnsPfcHor']
    RPfcHor = isttok_mag['RPfcHor']
    ZPfcHor = isttok_mag['ZPfcHor']
    IgainHor = isttok_mag['IgainHor']

    turnsP  = isttok_mag['TurnsPfcPrim']
    RPfcPrim = isttok_mag['RPfcPrim']
    ZPfcPrim = isttok_mag['ZPfcPrim']
    IgainPrim = isttok_mag['IgainPrim']

    Is2Bpol=np.zeros((nPrb, ns))
    #    Vertical Coils
    for k in range(len(turnsV)):
        br,bz= mf.Bloop(RPfcVer[k], ZPfcVer[k], Rprb, Zprb)
        bpol, brad = mf.BpolBrad(br,bz, tethaProb)
        Is2Bpol[:,0] += turnsV[k] * bpol * IgainVert

    #    Horizontal Coils
    for k in range(len(turnsH)):
        br,bz= mf.Bloop(RPfcHor[k], ZPfcHor[k], Rprb, Zprb)
        bpol, brad = mf.BpolBrad(br,bz, tethaProb)
        Is2Bpol[:,1] += turnsH[k] * bpol * IgainHor

    #    Primary Coils
    for k in range(len(turnsP)):
        br,bz= mf.Bloop(RPfcPrim[k], ZPfcPrim[k], Rprb, Zprb)
        bpol, brad = mf.BpolBrad(br,bz, tethaProb)
        Is2Bpol[:,2] += turnsP[k] * bpol*IgainPrim

    return Is2Bpol


def buildIc2Bpol(RIc, ZIc):
    """
    Build B poloidal response Matrix on the poloidal field probes from a set of filaments
    Gives poloidal field on each probe for a Is=1A on filament
    #

    Args:
    #        np: number of poloidal probes
    #        ns: number of PF

    Returns:
        MsBpol :0.0
    """
    nc   = len(RIc)
    nPrb = isttok_mag['nPrb']

    Rprb = isttok_mag['Rprb']
    Zprb = isttok_mag['Zprb']
    tethaProb = isttok_mag['tethaPrb']

    Ic2Bpol=np.zeros((nPrb, nc))
    #    each filamen
    for k in range(len(RIc)):
        br,bz= mf.Bloop(RIc[k], ZIc[k], Rprb, Zprb)
        bpol, brad = mf.BpolBrad(br,bz, tethaProb)
        Ic2Bpol[:,0] +=  bpol

    return Ic2Bpol
0.0
#        br,bz=Bloop(Rver[i], Zver[i], Rprb, zprb)
#        BR += Turns[i]*br
#        BZ += Turns[i]*bz


if __name__ == "__main__":

    #SDAS DATA
    client=StartSdas()
    shotP=44501
    #%matplotlib qt4
    times, mirnovs_P = getMirnovInt(client, shotP, 'Post')
    timesp,I_prim, tbs = getSignal(client, ch_prim, shotP )

    np.set_printoptions(precision=3)
    nc = 15 # number of coppe2r shell 'wires'

    ResCopper = 1.0e-4 # 0.1 mOhm
    aCopper = 10.0e-3  # 'wire' radius 10 mm

    Is2Bpol=buildIs2BpolB(isttok_mag)
    Is2Bpol1=buildIs2BpolB(isttok_mag_1)
    Is2Bpol2=buildIs2BpolB(isttok_mag_2)

    # make heaviside current signal
#    Imax = 157 #A
#    n1=np.int(0.2*FsamplingMARTe)
#    n2=np.int((0.6 -0.2 )*FsamplingMARTe)
#    n3=np.int((1.0 -0.6 )*FsamplingMARTe)
    #currPrim = np.concatenate([np.zeros(n1,), Imax*np.ones(n2,), np.zeros(n3,)])
    currPrim = I_prim[10:]
    currVert = np.zeros_like(currPrim)
    currHor=np.zeros_like(currPrim) # Zero current on Hori Field Coils

    IsPfc = np.array([currVert, currHor, currPrim])

    def allowedAngle(a):
        if a <20: return False
        elif a <70: return True
        elif a <110: return False
        elif a <175: return True
        elif a <180: return False
        elif a <265: return True
        elif a <275: return False
        elif a <340: return True
        else: return False
    # Copper passive 'filament ' positions
    segments=[[20, 70],[ 110, 175],[ 180,265],[ 275, 340]]
    angles = np.array([(i/(21*1.))*360 for i in range(21)])
    anglesIc = np.asarray([angle for angle in angles if allowedAngle(angle)])
    #anglesIc = np.array([(i/(nc*1.))*2*np.pi for i in range(nc)])
    anglesIc = np.radians(anglesIc)

    RIc = isttok_mag['RM'] + isttok_mag['Rcopper']  * np.cos(anglesIc)
    ZIc = isttok_mag['Rcopper']  * np.sin(anglesIc)

    plt.figure()
    plt.plot(RIc,ZIc,"*")
    plt.plot(isttok_mag['Rprb'],isttok_mag['Zprb'], "+")
    plt.plot(isttok_mag['Rprb'][2],isttok_mag['Zprb'][2], "o")

    A,Bs,C,Ds = buildLtiModelB(RIc, ZIc, isttok_mag)
    A1,Bs1,C1,Ds1 = buildLtiModelB(RIc, ZIc,isttok_mag_1)
    A2,Bs2,C2,Ds2 = buildLtiModelB(RIc, ZIc,isttok_mag_2)

    Ic2Bpol=buildIc2Bpol(RIc, ZIc)

    magSys = signal.StateSpace(A,Bs,C,Ds)
    magSys1 = signal.StateSpace(A1,Bs1,C1,Ds1)
    magSys2 = signal.StateSpace(A2,Bs2,C2,Ds2)

    #t,ic = signal.step(magSys)

    # Stability
    w, vect = np.linalg.eig(A)
    w1, vect1 = np.linalg.eig(A1)

    # Eigenvalues should all be negative
    print('LTI system Eigenvalues:')
    print(w)
    print(w1)

    bPolIs=np.matmul(Is2Bpol,IsPfc)
    bPolIs1=np.matmul(Is2Bpol1,IsPfc)
    bPolIs2=np.matmul(Is2Bpol2,IsPfc)
    #time = np.arange(IsPfc.shape[1]) / FsamplingMARTe
    time = times[10:]/times[-1] #trim negative times normalize


    # scipy.signal.lsim(system, U, T, X0=None, interp=True)
    #  U: If there are multiple inputs, then each column of the rank-2 array represents an input.

    tout, Ic, x = signal.lsim(magSys,IsPfc.T, time)
    tout1, Ic1, x1 = signal.lsim(magSys1,IsPfc.T, time)
    tout2, Ic2, x2 = signal.lsim(magSys2,IsPfc.T, time)

#    fig, ax = plt.subplots()
#    linesIc = ax.plot(time, Ic)
#    ax.legend(linesIc, ['ic0', 'ic1', 'ic2','ic3', 'ic4', 'ic5'], loc='best')
# #   lineIsv = ax.plot(time, IsPfc[0,:])
#    ax.set_xlabel('Time/s')0.0
#    plt.show()
#
    bPolIc=np.matmul(Ic2Bpol,Ic.T)
    bPolIc1=np.matmul(Ic2Bpol,Ic1.T)
    bPolIc2=np.matmul(Ic2Bpol,Ic2.T)

    bPolTot = (bPolIs.T +  bPolIc.T)*50*49e-6
    bPolTot1 = (bPolIs1.T +  bPolIc1.T)*50*49e-6
    bPolTot2 = (bPolIs2.T +  bPolIc2.T)*50*49e-6

    plotAllPoloid2(time, np.asarray(mirnovs_P)[:,10:]*1e6, bPolTot2.T*1e6, show=True, title='',  ylim=11)
'''
    for p in range(12):
        fig, ax = plt.subplots()
        linesMirnov= ax.plot(time, mirnovs_P[p][10:]*1e6, "-", label="Mirnov")
        #linesBpol = ax.plot(time, bPolTot[:,3]*1e6, ":", label="Original Positions")
        linesBpol1 = ax.plot(time, bPolTot1[:,p]*1e6, label="Optimized Positions")

        #ax.legend(linesBpol, ['m0', 'm1', 'm2','m3', 'm4', 'm5','m6', \
        #                         'm7', 'm8','m9', 'm10', 'm11'],loc='right')
        #ax.legend(linesBpol1, ['m0', 'm1', 'm2','m3', 'm4', 'm5','m6', \
        #                        'm7', 'm8','m9', 'm10', 'm11'],loc='best')
        plt.legend()
        ax.set_title('Probe response')
        ax.set_xlabel('Time/s')
        #ax.legend()
        plt.show()
%matplotlib qt4
'''

    #bPolIc=np.matmul(Is2Bpol,IsPfc)
#    BR = 0.0
#    BZ = 0.0
#    for c in range(len(isttok_mag['RPfcVer'])):
#        br,bz=mf.Bloop(isttok_mag['RPfcVer'][c], isttok_mag['ZPfcVer'][c], isttok_mag['Rprb'], isttok_mag['Zprb'])
#        BR += isttok_mag['TurnsPfcVer'][c]*br
#        BZ += isttok_mag['TurnsPfcVer'][c]*bz

#    plt.figure()
##    plt.plot(t2,y2,'g:',linewidth=2,label='State Space')
#    lineObjects = plt.plot(t,ic,linewidth=1)
#    plt.xlabel('Time/s')
#    plt.ylabel('Response ((ic) /A ')
#    plt.legend(lineObjects, ['ic0', 'ic1', 'ic2','ic3', 'ic4', 'ic5'],loc='best')
#    plt.show()

#    plt.figure()
#    #
#    lineObjs = plt.plot(t,np.matmul(ic,Bpolc) )
#    plt.xlabel('Time/s')
#    plt.ylabel('Bpol ')
#    plt.legend(lineObjs, ['m0', 'm1', 'm2','m3', 'm4', 'm5','m6', \
#                             'm7', 'm8','m9', 'm10', 'm11'],loc='right')
#    plt.show()
