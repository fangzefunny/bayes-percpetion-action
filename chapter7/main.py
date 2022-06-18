import numpy as np
from scipy.stats import norm 

import matplotlib.pyplot as plt 
import seaborn as sns

sns.set_context('talk')
sns.set_style("ticks", {'axes.grid': False})

Red = np.array([199, 111, 132]) / 255

def discriminate(k='Bayes', p_S=[.6, .4]):

    # s1 and s0
    Sp =  2; p_Sp = p_S[0]
    Sn = -1; p_Sn = p_S[1]
    sig = 2

    # discriminate
    if k == 'Bayes':
        k = (Sp+Sn)/2 - sig**2/(Sp-Sn)*np.log(p_Sp/p_Sn) 

    # p(s1|s1), p(s0|s1), p(s1|s0), p(s0|s0), 
    p_Sp1Sp = norm.cdf((Sp-k)/sig)
    p_Sn1Sp = 1 - p_Sp1Sp
    p_Sp1Sn = norm.cdf((Sn-k)/sig)
    p_Sn1Sn = 1 -  p_Sp1Sn

    return [p_Sp1Sp, p_Sn1Sp, p_Sp1Sn, p_Sn1Sn]

def confMatrix():

    # get Bayes discriminat 
    [p_Sp1Sp, p_Sn1Sp, p_Sp1Sn, p_Sn1Sn] = discriminate(k='Bayes') 

    # build confusion matrix 
    confMat = [[p_Sp1Sp, p_Sn1Sp],
               [p_Sp1Sn, p_Sn1Sn]]

    sns.heatmap(np.array(confMat), cmap='Blues', lw=1, square=True)
    plt.xticks([0.5, 1.5], ['S+', 'S-'])
    plt.yticks([0.5, 1.5], ['S+', 'S-'])
    plt.text(0.5-.15, 0.5+.1, f'{p_Sp1Sp:.2f}')
    plt.text(1.5-.15, 0.5+.1, f'{p_Sn1Sp:.2f}')
    plt.text(0.5-.15, 1.5+.1, f'{p_Sp1Sn:.2f}')
    plt.text(1.5-.15, 1.5+.1, f'{p_Sn1Sn:.2f}')
    plt.savefig('confMat.png', dpi=250)

def rocCurve():

    nK = 50
    ks = np.linspace(-5, 6, nK)
    hit = np.zeros([nK,]) + np.nan 
    fa  = np.zeros([nK,]) + np.nan
    for i, k in enumerate(ks):
        res = discriminate(k)
        hit[i] = res[0]
        fa[i]  = res[2]

    plt.figure(figsize=(5, 4.8))
    x = np.linspace(0, 1, 50)
    sns.lineplot(x=fa, y=hit, lw=3)
    plt.plot(x, x, ls='--', color='k')
    plt.plot(discriminate()[2], discriminate()[0], 'o', 
            color=Red, markersize=10)
    plt.text(discriminate()[2]+.05, discriminate()[0]-.02, 'p(s+)=.6')
    plt.plot(discriminate(p_S=[.5,.5])[2], 
             discriminate(p_S=[.5,.5])[0], 'ro', 
            color=Red, markersize=10)
    plt.text(discriminate(p_S=[.5,.5])[2]+.05, 
             discriminate(p_S=[.5,.5])[0]-.05, 'p(s+)=.5')
    plt.plot([discriminate(p_S=[.5,.5])[2], .5], 
             [discriminate(p_S=[.5,.5])[0], .5], lw=3, ls='--', color='orange')
    plt.plot([discriminate(p_S=[.5,.5])[2], discriminate(p_S=[.5,.5])[2]], 
             [0, discriminate(p_S=[.5,.5])[0]], lw=3, ls='--', color='orange')
    plt.plot([0, discriminate(p_S=[.5,.5])[2]], 
             [0, 0], lw=3, ls='--', color='orange')
    plt.xlabel('False alarm')
    plt.ylabel('Hit')
    plt.tight_layout()
    plt.savefig('roc.png', dpi=250)


if __name__ == '__main__':

    #confMatrix()
    rocCurve()