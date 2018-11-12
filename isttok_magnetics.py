# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 18:53:42 2018

@author: bernardo
"""

from __future__ import print_function
import numpy as np
import magnetic_flux_fields as mf

#ISTTOK
#
rm = 0.085 #Minor radius
RM = 0.46   # Major radius
Rmirn = 0.0935  #Mirnov Probe Radius   9.35

nPrb = 12
# Poloidal Angle of Mirnov probe
#tethaPrb = np.array([(23./24 - i/nPrb)*2*np.pi for i in range(nPrb)])
tethaPrb = np.array([(345.0  - i *30) for i in range(nPrb)]) /180.0  *np.pi

Rprb =RM + Rmirn * np.cos(tethaPrb)
Zprb =Rmirn * np.sin(tethaPrb)

#Vertical Coils: 4 coils, 5 turns, R1,2=58 [cm],R2,3=35 [cm],z=±7 [cm]
RPfcVer =[0.58, 0.58, 0.35, 0.35]
ZPfcVer =[0.07, -0.07, 0.07, -0.07]
TurnsPfcVer=[-5., -5., 5., 5.]
#Horizontal Coils: 2 coils , 4 turns, R1,2=58 [cm],z=±7[cm]
RPfcHor =[0.58, 0.58,]
ZPfcHor =[0.07, -0.07]
TurnsPfcHor=[4., -4.]

#Primary Coils: 2 coils , 14 turns, R1,2=62 [cm],z=±13[cm]
RPfcPrim =[0.62, 0.62,]
ZPfcPrim =[0.13, -0.13]
TurnsPfcPrim=[14., 14.]


isttok_mag = {'RM': RM, 'rm': rm, 'Rmirn': Rmirn , 'Rcopper': 0.105, 'nPrb':nPrb, \
          'tethaPrb':tethaPrb, 'Rprb':Rprb , 'Zprb':Zprb, \
          'RPfcVer':RPfcVer, 'ZPfcVer':ZPfcVer, 'TurnsPfcVer':TurnsPfcVer, \
          'RPfcHor':RPfcHor, 'ZPfcHor':ZPfcHor, 'TurnsPfcHor':TurnsPfcHor, \
          'RPfcPrim':RPfcPrim, 'ZPfcPrim':ZPfcPrim, 'TurnsPfcPrim':TurnsPfcPrim }

def toM (A):
    return [a*1e-2 for a in A]

#Optimized positions
#Vertical Coils: 4 coils, 5 turns, R1,2=58 [cm],R2,3=35 [cm],z=±7 [cm]

RPfcVer =toM([55.1,55.4,38.5,37.2])
ZPfcVer =toM([-13.2,16.7,-15.2,12.7])
TurnsPfcVer=[-5., -5., 5., 5.]

#Horizontal Coils: 2 coils , 4 turns, R1,2=58 [cm],z=±7[cm]
RPfcHor =toM([54.4,54.5])
ZPfcHor =toM([-10.8,13.4])
TurnsPfcHor=[4., -4.]

#Primary Coils: 2 coils , 14 turns, R1,2=62 [cm],z=±13[cm]
RPfcPrim =toM([61.5,61.5])
ZPfcPrim =toM([-14.4,14.5])
TurnsPfcPrim=[14., 14.]

isttok_mag_1 = {'RM': RM, 'rm': rm, 'Rmirn': Rmirn , 'Rcopper': 0.105, 'nPrb':nPrb, \
          'tethaPrb':tethaPrb, 'Rprb':Rprb , 'Zprb':Zprb, \
          'RPfcVer':RPfcVer, 'ZPfcVer':ZPfcVer, 'TurnsPfcVer':TurnsPfcVer, \
          'RPfcHor':RPfcHor, 'ZPfcHor':ZPfcHor, 'TurnsPfcHor':TurnsPfcHor, \
          'RPfcPrim':RPfcPrim, 'ZPfcPrim':ZPfcPrim, 'TurnsPfcPrim':TurnsPfcPrim }


def buildIs2Bpol(Vcoil, Hcoil=None, PrimCoil=None):
    """
    Build B poloidal response Matrix on the poloidal field probes from a set of
    PFC coil circuits (Vertical + Horizontal + Primary)
    Gives poloidal field on each probe for a Is=1A on coils
    #

    Args:
        Vcoil: numpy.array([RPfcVer,ZPfcVer,TurnsPfcVer])

    Returns:
        Is2Bpol : array [12, 3]
    """
    ns = 3 # number of PFC active independent coils circuits (sources)
#       number of poloidal probes
    nPrb = isttok_mag['nPrb']

    Rprb = isttok_mag['Rprb']
    Zprb = isttok_mag['Zprb']
    tethaProb = isttok_mag['tethaPrb']

    Is2Bpol=np.zeros((nPrb, ns))

#    Vertical Coils
    for k in range(Vcoil.shape[0]):
        br,bz= mf.Bloop(Vcoil[k,0], Vcoil[k,1], Rprb, Zprb)
        bpol, brad = mf.BpolBrad(br,bz, tethaProb)
        Is2Bpol[:,0] += Vcoil[k,2] * bpol

#    Horizontal Coils
    if isinstance(Hcoil, np.ndarray):
        for k in range(Hcoil.shape[0]):
            br,bz= mf.Bloop(Hcoil[k,0], Hcoil[k,1], Rprb, Zprb)
            bpol, brad = mf.BpolBrad(br,bz, tethaProb)
            Is2Bpol[:,1] += Hcoil[k,2] * bpol

#    Primary Coils
    if isinstance(PrimCoil, np.ndarray):
        for k in range(PrimCoil.shape[0]):
            br,bz= mf.Bloop(PrimCoil[k,0], PrimCoil[k,1], Rprb, Zprb)
            bpol, brad = mf.BpolBrad(br,bz, tethaProb)
            Is2Bpol[:,2] += PrimCoil[k,2] * bpol

    return Is2Bpol
