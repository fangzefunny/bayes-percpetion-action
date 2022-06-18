import os 
import numpy as np
import pandas as pd
from scipy.stats import norm 

from tqdm import tqdm

import matplotlib.pyplot as plt 
import seaborn as sns 


# find the current path
path = os.path.dirname(os.path.abspath(__file__))

#-------------------------------------
#         Visualize package 
#-------------------------------------

class viz:
    '''Define the default visualize configure
    '''
    Blue    = .95 * np.array([ 46, 107, 149]) / 255
    Green   = .95 * np.array([  0, 135, 149]) / 255
    Red     = .95 * np.array([199, 111, 132]) / 255
    Yellow  = .95 * np.array([220, 175, 106]) / 255
    Purple  = .95 * np.array([108,  92, 231]) / 255
    Palette = [Blue, Red, Green, Yellow, Purple]
    Greens  = [np.array([8,154,133]) / 255, np.array([118,193,202]) / 255] 
    dpi     = 200
    sfz, mfz, lfz = 11, 13, 16
    lw, mz  = 2.5, 6.5

    @staticmethod
    def get_style(): 
        sns.set_context('talk')
        sns.set_style("ticks", {'axes.grid': False})
        
viz.get_style()

#-------------------------------------
#         Riemannian integration
#-------------------------------------

normpdf = lambda x, mu, sig: np.exp(-.5*((x-mu)/sig)**2)/np.sqrt(2*np.pi)/sig

def riemanInt(p_same=.5, mu=2, sig=1.5, n_sample=51):
    '''Solve response distribution using the Riemannian Integration 

    Args:
        p_same: the probability of the condition "same", C=1
        mu: the signal
        sig: the measurement noisy

    Returns:
        p_Cp1Cp: p(CHat=1|C=1)
    '''
    # the discretize distribution, 
    x1 = x2 = np.linspace(-5, 5, n_sample)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    p_X1X2_p = normpdf(x1, mu, sig).reshape([1, -1])*normpdf(x2, mu, sig).reshape([-1, 1])
    p_X1X2_n = normpdf(x1, -mu, sig).reshape([1, -1])*normpdf(x2, -mu, sig).reshape([-1, 1])
    p_X1X2_same = .5 * (p_X1X2_p / p_X1X2_p.sum() + p_X1X2_n / p_X1X2_n.sum())

    # the inference criterion
    def d(x1, x2):
        e1 = np.exp(-mu*(x1+x2)/sig**2)
        e2 = np.exp( mu*(x1+x2)/sig**2)
        e3 = np.exp( mu*(x1-x2)/sig**2)
        e4 = np.exp(-mu*(x1-x2)/sig**2)
        return np.log((e1+e2)/(e3+e4)) + np.log(p_same/(1-p_same))

    # the response distribution
    f_X1X2 = np.array(list(map(d, list(x1_mesh), list(x2_mesh)))).reshape([n_sample, n_sample])
    res  = p_X1X2_same * (f_X1X2>0)
    res1 = p_X1X2_same * (f_X1X2<=0)
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), sharex=True, sharey=True)
    ax = axs[0]
    sns.heatmap(np.array(res), cmap='Blues', square=True, ax=ax)
    ax.invert_yaxis()
    ax.set_title(f'p("same"|"same")= {res.sum():.2f}\n')
    ax.set_xticks(range(0, n_sample, 10))
    ax.set_xticklabels(x1[range(0, n_sample, 10)].round(2), rotation=45)
    ax.set_yticks(range(0, n_sample, 10))
    ax.set_yticklabels(x1[range(0, n_sample, 10)].round(2), rotation=45)
    ax = axs[1]
    sns.heatmap(np.array(res1), cmap='Blues', square=True, ax=ax)
    ax.set_xticks(range(0, n_sample, 10))
    ax.set_xticklabels(x1[range(0, n_sample, 10)].round(2), rotation=45)
    ax.set_yticks(range(0, n_sample, 10))
    ax.set_yticklabels(x1[range(0, n_sample, 10)].round(2), rotation=45)
    ax.set_title(f'p("diff"|"same")= {res1.sum():.2f}\n')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{path}/rim-prior={p_same}.png', dpi=250)


normpdf = lambda x, mu, sig: np.exp(-.5*((x-mu)/sig)**2)/np.sqrt(2*np.pi)/sig

def MonteCarlo(p_same=.5, mu=2, sig=1.5, n_sample=100_000, nbins=100):
    '''Solve response distribution using the MonteCarlo method 

    Args:
        p_same: the probability of the condition "same", C=1
        mu: the signal
        sig: the measurement noisy
        n_sample: number of sample 
        nbins: bins for visualization

    Returns:
        res:  p(CHat=1|C=1)
        res1: p(CHat=2|C=1)
    '''

    # the inference criterion
    def d(x1, x2):
        e1 = np.exp(-mu*(x1+x2)/sig**2)
        e2 = np.exp( mu*(x1+x2)/sig**2)
        e3 = np.exp( mu*(x1-x2)/sig**2)
        e4 = np.exp(-mu*(x1-x2)/sig**2)
        return np.log((e1+e2)/(e3+e4)) + np.log(p_same/(1-p_same))
    
    # sample
    X1_p = norm(mu, sig).rvs(size=n_sample)
    X2_p = norm(mu, sig).rvs(size=n_sample)
    f_X1X2_p = np.array(list(map(d, X1_p, X2_p)))
    X1_n = norm(-mu, sig).rvs(size=n_sample)
    X2_n = norm(-mu, sig).rvs(size=n_sample)
    f_X1X2_n = np.array(list(map(d, X1_n, X2_n)))
    
    # the response distribution
    res  = ((f_X1X2_p>0).sum()+(f_X1X2_n>0).sum())/(n_sample*2) # p(same|same)
    res1 = ((f_X1X2_p<0).sum()+(f_X1X2_n<0).sum())/(n_sample*2) # p(false|same)

    ## for visualization 
    x_bins = y_bins = np.linspace(-7, 7, nbins)
    x1_mesh, x2_mesh = np.meshgrid(x_bins, y_bins)
    x1_sample = np.hstack([X1_p, X1_n])
    x2_sample = np.hstack([X2_p, X2_n])
    X1_discret = np.digitize(x1_sample, x_bins)
    X2_discret = np.digitize(x2_sample, y_bins) 
    mat = np.zeros([nbins, nbins])
    for i, j in zip(X1_discret, X2_discret):
        if (i<nbins) and (j<nbins): mat[i, j] += 1
    p_X1X2 = mat / mat.sum()
    f_X1X2 = np.array(list(map(d, list(x1_mesh), list(x2_mesh)))).reshape([nbins, nbins])

    ## visualize 
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.7))
    ax = axs[0]
    sns.heatmap(np.array(p_X1X2*(f_X1X2>0)), cmap='Blues', square=True, ax=ax)
    ax.invert_yaxis()
    ax.set_title(f'p("same"|"same")= {res:.2f}\n')
    ax.set_xticks(range(0, nbins, 15))
    ax.set_xticklabels(x_bins[range(0, nbins, 15)].round(2), rotation=45)
    ax.set_yticks(range(0, nbins, 15))
    ax.set_yticklabels(y_bins[range(0, nbins, 15)].round(2), rotation=45)
    ax = axs[1]
    ax.scatter(x=x1_sample, y=x2_sample, s=2, color='k')
    ax.set_xticks(x_bins[range(0, nbins, 15)].round(2))
    ax.set_xticklabels(x_bins[range(0, nbins, 15)].round(2), rotation=45)
    ax.set_yticks(x_bins[range(0, nbins, 15)].round(2))
    ax.set_yticklabels(y_bins[range(0, nbins, 15)].round(2), rotation=45)
    ax.set_title(f'samples\n')
    ax.axis('square')
    plt.tight_layout()
    plt.savefig(f'{path}/mc-prior={p_same}.png', dpi=250)

if __name__ == '__main__':

    MonteCarlo(p_same=.75)

