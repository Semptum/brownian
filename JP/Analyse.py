import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import scipy.stats as st


def init(filename):
    file=sio.loadmat(filename) #On charge le fichier
    Nbobjets = int(max(file['objs_link'][3]))
    print("On a tracqué "+str(Nbobjets)+" particules")
    return file,Nbobjets

def take(i,binning,agrandissement,frequence):
    """La fonction qui tracque la particule i"""
    PixelSize=binning*(6450/agrandissement)*1e-3 # en um
    x1=file['objs_link'][0,:][np.where(file['objs_link'][5,:]==i)]*PixelSize
    y1 = file['objs_link'][1,:][np.where(file['objs_link'][5,:]==i)]*PixelSize
    dureetrajectoire=x1.shape[0]
    print("Trajectory acquired for particle "+str(i))
    T=np.arange(dureetrajectoire)/frequence
    if dureetrajectoire>0:
        return x1-x1[0],y1-y1[0],T,dureetrajectoire
    else:
        return [0],[0],T,dureetrajectoire

def plot_dx_amp(X,Y):
    t = np.linspace(0,1,X.shape[0])
    dx=X[1:]-X[:-1]
    dy=Y[1:]-Y[:-1]
    dr=(dx**2+dy**2)**0.5
    dr/=dr.max()
    points = np.array([X,Y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap('jet'))
    lc.set_array(dr)
    plt.gca().add_collection(lc)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())

def autocorr(x,formsd=False):
    fftx = np.fft.fft(x, n=(x.shape[0]*2-1), axis=0)
    cfftx=np.conjugate(fftx)
    ret = np.real(np.fft.ifft(fftx*cfftx, axis=0))
    result = np.fft.fftshift(ret, axes=0)
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


def GaussianHisto(dx,n=30):
    """
    Fonction qui calcule l'histogramme supposé gaussien et le fite
    """
    a,b=np.histogram(dx,n)
    ok=a>0
    x,y=((b[1:]+b[:-1])/2)[np.where(ok)],np.log(a[np.where(ok)])
    p = np.poly1d(np.polyfit(x,y,2,w=y))
    if p[2]>0:
        p[2]=-np.inf
    return x,y,p
def sigma(p):
    return (-1/p[2]/2)**0.5

def sigma2fromtau(X,Y,t,NbrTau=100):
    length=X.shape[0]
    assert length>10, "pas assez de pts"
    TauMax=length-100
    Tau=np.linspace(0,TauMax,NbrTau)
    S2=np.zeros_like(Tau)
    for i in range(Tau.shape[0]):
        tau=int(Tau[i])
        mux, sx = st.norm.fit(X[tau+1:]-X[:-tau-1])
        mux, sy = st.norm.fit(Y[tau+1:]-Y[:-tau-1])
        S2[i]=sx**2+sy**2
    Tau/=Tau.max()
    Tau*=t[TauMax]
    return Tau,S2

def deltax(r):
    DeltaX=np.zeros((N,N))
    for i in range(N):
        DeltaX[i,:N-i-1]=r[i+1:]-r[:-i-1]
    return DeltaX

def truedelta(r,tau):
    return r[tau::tau]-r[:-tau:tau]

def test(k,X,Y,r0=0.1,sliding=True):
    """
    Fonction qui calcule la repartition des proba de position apres k pas en utilisant une echelle spatiale de r0
    """
    if sliding:
        dx=X[k+1:]-X[:-k-1]
        dy=Y[k+1:]-Y[:-k-1]
    else:
        dx=truedelta(X,k+1)
        dy=truedelta(Y,k+1)
    d=(dx**2+dy**2)**0.5
    n=int(d.max()/r0)
    return np.histogram(d,n)

def positionmoyenne(k,X,Y,r0=0.1,sq=1,sliding=True):
    """
    Calcule l'esperance de la distance du départ en fonction du nombre de pas
    """
    a,b=test(k,X,Y,r0,sliding)
    x=(b[:-1]+b[1:])/2
    return np.average(x**sq,weights=a)

def diffusion(sigma2):
    return sigma2/4*frequence
def viscosity(D):
    return kb*temp/(6*np.pi*diametre/2*D)

def concatenate(ilist,binning,agrandissement,frequence,minlength):
    X=[0]
    Y=[0]
    N=0
    for i in ilist:
        a=take(i,binning,agrandissement,frequence)
        if a[-1]>minlength:
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
