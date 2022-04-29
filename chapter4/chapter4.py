import os 
import numpy as np
import pandas as pd
from scipy.stats import norm 

from tqdm import tqdm
from concurrent.futures  import ThreadPoolExecutor

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
    dpi     = 200
    sfz, mfz, lfz = 11, 13, 16
    lw, mz  = 2.5, 6.5

    @staticmethod
    def get_style(): sns.set_style("ticks", {'axes.grid': False})

#-------------------------------------
#            Section 4.5 
#-------------------------------------

def Fig4_3(s_check=10, param_Prior=(0, 8), param_Like=4, 
                n_samples=100000, n_bins=60, seed=1234):

    # fix random seed
    rng = np.random.RandomState(seed)

    # get distribution parameters 
    mu, sig_s = param_Prior
    sig       = param_Like
    J, J_s    = 1/sig**2, 1/sig_s**2
    sig_res   = np.sqrt(J/(J+J_s)**2) 

    # get PM prediction 
    hat_PM = lambda s: [norm((J*si+J_s*mu)/(J+J_s), sig_res
                        ).rvs(random_state=rng) for si in tqdm(s, ncols=100)]
    # get ML prediction 
    hat_ML = lambda s:[norm(si, sig).rvs(random_state=rng)
                        for si in tqdm(s, ncols=100)]

    # get s the real stimulus and prediction 
    s = norm(mu, sig_s).rvs(size=n_samples, random_state=rng)
    
    # get two threads
    pool = ThreadPoolExecutor(max_workers=2)
    job_ML = pool.submit(hat_ML, s)
    job_PM = pool.submit(hat_PM, s)
    shat_ML = job_ML.result()
    shat_PM = job_PM.result()
    pool.shutdown()

    # construct a dataframe for plotting 
    bins = np.linspace(-30, 30, n_bins)
    data = pd.DataFrame(np.vstack([s, shat_ML, shat_PM]).T, 
                        columns=['s', 'shat_ML', 'shat_PM'])
    data['s_digit']  = np.digitize(data['s'], bins=bins)
    data['ML_digit'] = np.digitize(data['shat_ML'], bins=bins) 
    data['PM_digit'] = np.digitize(data['shat_PM'], bins=bins) 
    data['s-ML_hat'] = data['shat_ML'] - data['s']
    data['s-ML_digit'] = np.digitize(data['s-ML_hat'], bins=bins) 
    data['s-PM_hat'] = data['shat_PM'] - data['s']
    data['s-PM_digit'] = np.digitize(data['s-PM_hat'], bins=bins) 

    # replicat figure 4.3 
    nr, nc = 2, 4
    fig, axs = plt.subplots(nr, nc, figsize=(3*nc, 2.9*nr))
    kws = ['ML', 'PM']
    for i, kw in enumerate(kws):
        ax = axs[i, 0]
        sns.scatterplot(x='s', y=f'shat_{kw}', data=data.loc[:1000, :], 
                color=viz.Palette[i], s=5, ax=ax)
        sns.lineplot(x=bins , y=bins ,
                ls='--', color=viz.Palette[i], ax=ax)
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        # infer p(shat|s=10)
        ax = axs[i, 1]
        sns.histplot(x=f'{kw}_digit', data=data.query(
            f's_digit=={np.digitize(s_check, bins=bins)}'),
            color=[.7]*3, edgecolor=[1,1,1], binwidth=1, 
            ax=ax)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xticklabels([-20, -10, 0, 10, 20])
        # infer p(s|shat=10)
        ax = axs[i, 2]
        sns.histplot(x=f's_digit', data=data.query(
            f'{kw}_digit=={np.digitize(s_check, bins=bins)}'), 
            color=[.3]*3, edgecolor=[1,1,1], binwidth=1, 
            ax=ax)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xticklabels([-20, -10, 0, 10, 20])
        # show prediction error distribution
        ax = axs[i, 3]
        sns.histplot(x=f's-{kw}_digit', data=data, 
            color=viz.Palette[i], edgecolor=[1,1,1], binwidth=1,
            ax=ax)
        ax.set_xticks([10, 20, 30, 40, 50])
        ax.set_xticklabels([-20, -10, 0, 10, 20])
    fig.tight_layout()
    plt.savefig(f'{path}/Fig_4_3_sig={param_Like}.png', dpi=viz.dpi)

if __name__ == '__main__':

    Fig4_3(param_Like=(8), n_samples=1000000)
        

