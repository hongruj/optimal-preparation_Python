import numpy as np
import matplotlib.pyplot as plt

def plot_eig(W, label="$W$", figsize=(7,5)):
    plt.figure(figsize=figsize)
    W_evals, _ = np.linalg.eig(W)
    plt.plot(W_evals.real,W_evals.imag,'o',color='grey')
    plt.axis([-11,5,-5,5])
    plt.xticks(np.arange(-11, 5, 1))
    plt.yticks(np.arange(-5, 5, 1))
    plt.ylabel("imag. $\lambda$")
    plt.xlabel("real $\lambda$")
    plt.text(3,4,"unstable",fontsize=22)
    plt.text(-1,4,"stable",fontsize=22,horizontalalignment="right")
    plt.axvline(x=1,linewidth=3, color='k', linestyle="--")
    plt.show()

def plot_W(W):
    plt.figure(figsize=(5,5))
    im = plt.imshow(W,cmap=plt.cm.bwr)
    n = W.shape[0]
    plt.xlim(0,n+1)
    plt.ylim(n+1,0)
    plt.yticks(np.arange(0,n+1,n/2))
    plt.yticks(np.arange(0,n+1,n/2))
    plt.clim(-2,2)
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.show()
    
def plot_energy(evals):
    plt.figure(figsize=(5,5))
    plt.ylabel("energy evoked $\epsilon$")
    plt.xlabel("$n^{th}$ eigenvector of $Q$")
    plt.plot(evals , "k")
    plt.title("original")
    plt.show()

def set_figure(cond):
    figsize=(20,10)
    ylim = 50
    if cond>4:
        fig, axes = plt.subplots(2,cond//2, figsize=figsize, sharey=True)
        axes = axes.flatten()
        axes[4].set_ylabel("firing rate [a.u.]")
        axes[4].set_yticks(np.arange(0,ylim+1,10))
    else:
        fig, axes = plt.subplots(1,cond, figsize=figsize, sharey=True)
    axes[0].set_ylabel("firing rate [a.u.]")
    axes[0].set_yticks(np.arange(0,ylim+1,10))
    return axes

def plot_fr_per_cond(axes, k, T, r, xstars, test=False):
    ylim = 60
    axes[k].set_ylim(0, ylim)
    axes[k].set_xlabel("t [s]")
    axes[k].set_xticks(np.arange(0, T+.1, 0.2))
    axes[k].text(0.3,0.9, "condition %i" % (k+1), transform=axes[k].transAxes, fontsize=20)
    idxs = range(0,r.shape[1],10)
    colors=iter(plt.cm.RdBu_r(np.linspace(0,1,len(idxs))))
    for j in idxs:
        c=next(colors)
        axes[k].plot(np.linspace(0,T,r.shape[0]),r[:,j],c=c)
        if test:
            axes[k].plot(T, xstars[j,k], marker='o', c=c, markersize=5)
            
def plot_out(out):
    figsize=(10,4.5)
    fig, axes = plt.subplots(2,4, figsize=figsize, sharey=True)
    axes = axes.ravel()
    for j in range(8):        
        axes[j].plot(out[j,0,:])
        axes[j].plot(out[j,1,:])       