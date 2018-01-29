#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 2018
Is a small script for analysing experimentally observed brownian motion of colloidal particules

@author: Andre Kalouguine & Laura Guislain & Nicolas Lecoeur
"""
import os
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


Folder="/home/andre/Documents/Labo/20180123/nico_1/"
File="TrackGUIoutput.mat"

os.chdir(Folder)
file=sio.loadmat(File)    

Nblink = max(file['objs_link'][5])
Nbobjets = max(file['objs_link'][3])
longueurminimale=int(1e3)

deltaglobal=np.zeros(longueurminimale)
deltamoyenglobal=np.zeros(longueurminimale)

binning=1
agrandissement=20
PixelSize=binning*(6450/agrandissement)*1e-3 # en um
frequence=400

i=1

def take(i):
    x1=file['objs_link'][0,:][np.where(file['objs_link'][5,:]==i)]
    dureetrajectoire=x1.shape[0]
    if dureetrajectoire > longueurminimale:
        x1*=PixelSize
        y1 = file['objs_link'][1,:][np.where(file['objs_link'][5,:]==i)]*PixelSize
        print("Trajectory acquired")
        T=np.arange(dureetrajectoire)/frequence
        return x1,y1,T,dureetrajectoire


def xcorr(x):
  fftx = np.fft.fft(x, n=(x.shape[0]*2-1), axis=0)
  cfftx=np.conjugate(fftx)
  ret = np.real(np.fft.ifft(np.array([fftx[i].dot(cfftx[i]) for i in range(fftx.shape[0])]), axis=0))
  ret = np.fft.fftshift(ret, axes=0)
  return ret[ret.size//2:]


def autocorr(x,formsd=False):
    result = np.correlate(x, x, mode='full')
    if formsd:
        N=result.size//2+1
        return result[result.size//2:]/(N-np.arange(0,N))
    return result[result.size//2:]


def msd(r):
  N=r.shape[0]
  D=r**2
  D=np.append(D,0)
  S2=autocorr(r,True)
  Q=2*D.sum()
  S1=np.zeros(N)
  for m in range(N):
      Q=Q-D[m-1]-D[N-m]
      S1[m]=Q/(N-m)
  return np.arange(N)/frequence,S1-2*S2

def deltax(r):
    DeltaX=np.zeros((N,N))
    for i in range(N):
        DeltaX[i,:N-i-1]=r[i+1:]-r[:-i-1]
    return DeltaX

def GaussianHisto(dx,n=30):
    """
    Fonction qui calcule l'histogramme supposé gaussien et le fite
    """
    a,b=np.histogram(dx,n)
    x,y=(b[1:]+b[:-1])/2,np.log(a+0.1)
    p = np.poly1d(np.polyfit(x,y,2,w=(y>1)*y))
    return x,y,p

def slowcorr(x,mul=np.dot):
    """
    Fonction inutile qui calcule la correlation. Présente pour archivage
    """
    n=x.shape[0]
    R=np.zeros(n)
    for j in range(n):
        arr=[mul(x[k],x[k-j]) for k in range(j,n)]
        R[j]=sum(arr)
    return R



def test(k,X=None,Y=None,r0=0.1):
    """
    Fonction qui calcule la repartition des proba de position apres k pas en utilisant une echelle spatiale de r0 
    """
    if X!=None:
        dkX=X[k+1:]-X[:-k-1]
    else:
        dkX=dx[k][:-k-1]
    if Y!=None:
        dkY=Y[k+1:]-Y[:-k-1]
    else:
        dkY=dy[k][:-k-1]

    ddr=np.array([dkX,dkY]).T
    x,y=ddr[:,0],ddr[:,1]
    d=(x**2+y**2)**0.5
    n=int(d.max()/r0)
    return np.histogram(d,n)

def positionmoyenne(k,X=None,Y=None,r0=0.1,sq=1):
    """
    Calcule l'esperance de la distance du départ en fonction du nombre de pas
    """
    a,b=test(k,X,Y,r0)
    x=(b[:-1]+b[1:])/2
    return np.average(x**sq,weights=a)

def hist2D(k,n=100):
    plt.hist2d(dx[k][:-k-1],dy[k][:-k-1],n)


def concatenate(ilist=np.arange(Nbobjets)+1):
    X=[0]
    Y=[0]
    N=0
    for i in ilist:
        a=take(i)
        if a!=None:
            x1,y1,t,n=a
            X+=list(x1-x1[0]+X[-1])[1:]
            Y+=list(y1-y1[0]+Y[-1])[1:]
            N+=n
    T=np.arange(N)/frequence
    return np.array(X),np.array(Y),T,N

def distmoy(i,kmax=1000):
    x1,y1,T,N=take(i)
    D=[]
    for i in range(kmax):
        D+=[positionmoyenne(i,x1,y1,sq=2)]
    D1=np.array(D)
    a,D2=msd((x1**2+y1**2)**0.5)
    plt.plot(D1**2)
    plt.plot(D2**2)
    
    
#dr=np.array([dx[0],dy[0]]).T

#X,Y,T,N=concatenate()
