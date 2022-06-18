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
#            Section 4.5 
#-------------------------------------

# closed-form solution of var and bias 
eps_ = 1e-12 # avoid divde by 0 
Var_ML   = lambda J, J_s: 1 / J  
Bias2_ML = lambda J, J_s, mu, s: 0 
Var_PM   = lambda J, J_s: J / ((J+J_s)**2+eps_)
Bias2_PM = lambda J, J_s, mu, s: (J_s**2*(mu - s)**2) / ((J+J_s)**2+eps_)

def Fig4_1(param_Prior=(0, 8), seed=1234):

    # instantiate a random generator
    rng = np.random.RandomState(seed)

    # get distribution parameters
    kws       = ['ML', 'PM'] 
    mu, sig_s = param_Prior
    J_s       = 1/(sig_s**2+eps_)
    sig_lst   = np.linspace(.01, 16, 30)
    s_lst     = np.linspace(-30, 30, 50)
    f_S       = norm(mu, sig_s).pdf(s_lst)
    p_S       = f_S / f_S.sum()
    
    # get data
    var_data   = np.zeros([2, len(sig_lst)]) + np.nan 
    bias2_data = np.zeros([2, len(sig_lst)]) + np.nan 
    for i, sig in enumerate(sig_lst):
        j = 1/sig**2
        for k, kw in enumerate(kws):
            var_data[k, i] = eval(f'Var_{kw}')(j, J_s)
            bias2 = np.array([eval(f'Bias2_{kw}')(j, J_s, mu, s) for s in s_lst])
            bias2_data[k, i] = (bias2*p_S).sum()
    mse_data = bias2_data + var_data
        
    # plotting 
    nr, nc = 2, 2
    fig, axs = plt.subplots(nr, nc, figsize=(4*nc, 4*nr), sharex=True, sharey=True)
    ax = axs[0, 0]
    for i in range(len(kws)):
        sns.lineplot(x=sig_lst, y=mse_data[i, :], color=viz.Greens[i],
        lw=4, ax=ax)
    ax.set_xlabel('Sigma')
    ax.set_title(r'Overall MSE')
    ax.legend(kws)
    ax = axs[0, 1]
    ax.set_axis_off()
    ax = axs[1, 0]
    for i in range(len(kws)):
        sns.lineplot(x=sig_lst, y=mse_data[i, :]-var_data[i, :], color=viz.Greens[i],
        lw=4, ax=ax)
    ax.set_xlabel('Sigma')
    ax.set_title('Overall Bias')
    fig.tight_layout()
    ax = axs[1, 1]
    for i in range(len(kws)):
        sns.lineplot(x=sig_lst, y=var_data[i, :], color=viz.Greens[i],
        lw=4, ax=ax)
    ax.set_xlabel('Sigma')
    ax.set_title('Var')
    fig.tight_layout()
    plt.savefig(f'{path}/Fig_4_1.png', dpi=viz.dpi)
    
#-------------------------------------
#           Section 4.5.1 
#-------------------------------------

def Fig4_3(param_Prior=(0, 8), param_Like=4, 
            n_samples=100000, n_bins=60, seed=1234):

    # instantiate a random generator
    rng = np.random.RandomState(seed)

    # get distribution parameters 
    mu, sig_s = param_Prior
    sig       = param_Like
    J, J_s    = 1/sig**2, 1/sig_s**2
    sig_res   = np.sqrt(J/(J+J_s)**2) 
    bins = np.linspace(-30, 30, n_bins)

    # get PM prediction 
    hat_PM = lambda s: [norm((J*si+J_s*mu)/(J+J_s), sig_res
                        ).rvs(random_state=rng) for si in tqdm(s, ncols=100)]
    # get ML prediction 
    hat_ML = lambda s:[norm(si, sig).rvs(random_state=rng)
                        for si in tqdm(s, ncols=100)]

    # get the sample data, load the cached if data exists
    fname = f'{path}/data-sig={param_Like}.csv'
    if os.path.exists(fname): data = pd.read_csv(fname)
    else: 
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
        data = pd.DataFrame(np.vstack([s, shat_ML, shat_PM]).T, 
                            columns=['s', 'shat_ML', 'shat_PM'])
        data['s_digit']  = np.digitize(data['s'], bins=bins)
        data['ML_digit'] = np.digitize(data['shat_ML'], bins=bins) 
        data['PM_digit'] = np.digitize(data['shat_PM'], bins=bins) 
        data['s-ML_hat'] = data['shat_ML'] - data['s']
        data['s-ML_digit'] = np.digitize(data['s-ML_hat'], bins=bins) 
        data['s-PM_hat'] = data['shat_PM'] - data['s']
        data['s-PM_digit'] = np.digitize(data['s-PM_hat'], bins=bins)

        # save data 
        data.to_csv(fname, index=False) 
    
    # infer p(shat|s=10)
    nr, nc = 4, 2
    fig, axs = plt.subplots(nr, nc, sharey='row',
                figsize=(3.2*nc, 3.*nr))
    kws   = ['ML', 'PM']
    svals = [1, 10]
    for i, kw in enumerate(kws):
        ax = axs[0, i]
        sns.scatterplot(x='s', y=f'shat_{kw}', data=data.loc[:1000, :], 
                color=viz.Greens[i], s=5, ax=ax)
        sns.lineplot(x=bins , y=bins ,
                ls='--', color=viz.Greens[i], ax=ax)
        ax.axvline(x=svals[0], ls='--', color=[.7]*3)
        ax.axvline(x=svals[1], ls='--', color=[.5]*3)
        ax.set_xlim([-30, 30])
        ax.set_ylim([-30, 30])
        ax.set_ylabel(f'Est.')
        ax.set_xlabel('s')

        ax = axs[1, i]
        sns.histplot(x=f'{kw}_digit', data=data.query(
            f's_digit=={np.digitize(svals[0], bins=bins)}'),
            color=[.7]*3, edgecolor=[1,1,1], binwidth=1, 
            stat='probability', ax=ax)
        ax.axvline(x=np.digitize(svals[0], bins=bins), ls='--', color=viz.Greens[i])
        ax.set_xlabel(f's={svals[0]}, common case')
        ax.set_ylabel(f'p(shat|s={svals[0]})')
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_xticklabels([30, -20, -10, 0, 10, 20, 30])

        # infer p(s|shat=10)
        ax = axs[2, i]
        sns.histplot(x=f'{kw}_digit', data=data.query(
            f's_digit=={np.digitize(svals[1], bins=bins)}'), 
            color=[.5]*3, edgecolor=[1,1,1], binwidth=1, 
            stat='probability', ax=ax)
        ax.axvline(x=np.digitize(svals[1], bins=bins), ls='--', color=viz.Greens[i])
        ax.set_xlabel(f's={svals[1]}, rare case')
        ax.set_ylabel(f'p(shat|s={svals[1]})')
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_xticklabels([30, -20, -10, 0, 10, 20, 30])

        # show prediction error distribution
        ax = axs[3, i]
        sns.histplot(x=f's-{kw}_digit', data=data, 
            color=viz.Greens[i], edgecolor=[1,1,1], binwidth=1,
            stat='probability', ax=ax)
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_xticklabels([30, -20, -10, 0, 10, 20, 30])
        ax.set_xlabel('s - shat')
    fig.tight_layout()
    plt.savefig(f'{path}/Fig_4_3_sig={param_Like}.png', dpi=viz.dpi)

if __name__ == '__main__':

    Fig4_1()
    Fig4_3(param_Like=(4), n_samples=1000000)
        

