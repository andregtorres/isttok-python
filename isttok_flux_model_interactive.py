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
import keyboard

def buildLtiModelB(RIc, ZIc, isttok_mag, Rc):
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
    #Rc = np.diag([1.0]*nc) * ResCopper

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

#FOR TWO LINES
def plotAllPoloid2(times_, dataArr1, dataArr2, show=True, title='',  ylim=0.0):
    """
    PLOTS ALL DATA FROM MIRNOVS in a poloidal arragment similar to Mirnov positions
    Args

    """
    fig, axs = plt.subplots(4, 4, sharex=True)
    plt.tight_layout()
    coilNr=0
    fig.suptitle(title)
   # ax=[]
    #ylim=2.0e6 # Y Axis limit
    #pltOrder = (11, )
    pltRow =    (2, 3,3,3,3, 2 , 1, 0,0,0,0, 1 )
    pltColumn = (3, 3,2,1,0, 0 , 0, 0,1,2,3, 3 )
   # pltColumn = (11, )
    axs[0,0].set_title('8')
    axs[0,3].set_title('11')
    axs[1,1].axis('off')
    axs[1,2].axis('off')
    axs[2,2].axis('off')
    axs[2,1].axis('off')

    for i in range(dataArr1.shape[0]):
        ax=axs[pltRow[coilNr], pltColumn[coilNr]]
        ax.plot(times_*1e-3, dataArr1[i,:])
        ax.plot(times_*1e-3, dataArr2[i,:])
        ax.ticklabel_format(style='sci',axis='y', scilimits=(0,0))
        ax.grid(True)
        if ylim >0.0:
            ax.set_ylim([-ylim, ylim])
        coilNr+=1

    if show:
        plt.show()
    #return fig,ax, line2


def fullModel(IsPfc, angles, Rc, time):
    Is2Bpol2=buildIs2BpolB(isttok_mag_2)
    RIc = isttok_mag['RM'] + isttok_mag['Rcopper']  * np.cos(anglesIc)
    ZIc = isttok_mag['Rcopper']  * np.sin(anglesIc)
    A2,Bs2,C2,Ds2 = buildLtiModelB(RIc, ZIc,isttok_mag_2, Rc)
    Ic2Bpol=buildIc2Bpol(RIc, ZIc)
    magSys2 = signal.StateSpace(A2,Bs2,C2,Ds2)
    bPolIs2=np.matmul(Is2Bpol2,IsPfc)
    tout2, Ic2, x2 = signal.lsim(magSys2,IsPfc.T, time)
    bPolIc2=np.matmul(Ic2Bpol,Ic2.T)
    bPolTot2 = (bPolIs2.T +  bPolIc2.T)*50*49e-6
    return bPolTot2, (RIc,ZIc)

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

    time = times[10:]  #trim negative
    timeN=time/times[-1] # times normalize

    # diagonal Resistance matrix Rc
    Rc = np.diag([1.0]*nc) * ResCopper
    bPolTot2, IcPositions =fullModel(IsPfc, anglesIc, Rc, timeN)

    print("START")
    plt.ion()
    fig1=plt.figure()
    ax1=fig1.add_subplot(111)
    plt.tight_layout()
    line11,=ax1.plot(isttok_mag['Rprb'],isttok_mag['Zprb'], "+")
    line12,=ax1.plot(IcPositions[0], IcPositions[1],"o")
    line13,=ax1.plot(IcPositions[0][0], IcPositions[1][0],"og")

    fig2=[]
    ax2=[]
    line21=[]
    line22=[]
    dataArr1=np.asarray(mirnovs_P)[:,10:]*1e6
    dataArr2=bPolTot2.T*1e6
    for i in range(12):
        fig2.append(plt.figure())
        ax2.append(fig2[i].add_subplot(111))
        plt.tight_layout()
        l21,=ax2[i].plot(time*1e-3, dataArr1[i,:])
        l22,=ax2[i].plot(time*1e-3, dataArr2[i,:])
        line22.append(l22)


    #plotAllPoloid2(time, np.asarray(mirnovs_P)[:,10:]*1e6, bPolTot2.T*1e6, show=True, title='',  ylim=11)

    fil=0
    while(1):
        key=keyboard.read_key()
        if key == "q":
            break
        elif key=="w":
            print("")
            print (fil)
            print (np.degrees(anglesIc))
        elif key=="a":
            if fil <14:
                fil+=1
        elif key=="z":
            if fil >0:
                fil-=1
        elif key=="s":
            anglesIc[fil]+=0.01
        elif key=="x":
            anglesIc[fil]-=0.01
        #elif key=="e":
            #plotAllPoloid2(time, np.asarray(mirnovs_P)[:,10:]*1e6, bPolTot2.T*1e6, show=True, title='',  ylim=11)

        bPolTot2, IcPositions =fullModel(IsPfc, anglesIc, Rc, timeN)

        line12.set_xdata(IcPositions[0])
        line12.set_ydata(IcPositions[1])
        line13.set_xdata(IcPositions[0][fil])
        line13.set_ydata(IcPositions[1][fil])

        dataArr2=bPolTot2.T*1e6
        for i in range(12):
            line22[i].set_ydata(dataArr2[i,:])
            fig2[i].canvas.draw()
            fig2[i].canvas.flush_events()

        fig1.canvas.draw()
        fig1.canvas.flush_events()
