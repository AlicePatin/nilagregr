# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 15:58:20 2022

@author: licep
"""

# IMPORT LIBRARIES
import numpy as np
import nigsp
import csv
import os
from copy import deepcopy

import phys2cvr as p2c




def load_func (fname):
    """
    Return the % of change and the spatial average of the function.

    Parameters
    ----------
    fname : str or path to a text file
       Fullpath/Filename of the functional input (txt). Time should be in first dimension.
       
    Returns
    -------
    func : ndarray
        Array containing the % of change of the timeseries, with time at the last dimension.
    func_avg : ndarray
        Array containing the spatial average of the function.

    """
    func = np.genfromtxt(fname_func, delimiter=' ')  #If we put ',' as delimiter, we have only one dimension
    func = func.T
    func_avg = func.mean(axis=0) # even if we have parcellation. Like this we have only one dimension.
    func = p2c.signal.spc(func)  # Compute signal % change
    return func, func_avg

# mettre atlas en optional, faire une solution si il n'y a pas d'atlas
# verifier que results.shape[1] = nb de regressors
# regarder quel est le plus energiquement mieux pour l'inclusion des boucles
# comment on choisit les valeurs newdim

def export_results (results, results_labels, atlas=None):
    # Export Beta, Tstat, Lag with/and with unfolding for EACH REG
    nb_regressors = results.shape[1]
    nb_res = len(results_labels)
    newdim = deepcopy(img_.header['dim'])
    newdim[0], newdim[4] = 3, 1 
    oimg = deepcopy(img_)
    oimg.header['dim'] = newdim 
    if (atlas != None):
        results_labels = ['atlas_'+x for x in results_labels]
    for res in range(nb_res):
        # Export all Beta, Tstat, Lag in csv
        np.savetxt(f'{fname_out_func}_{results_labels[res]}_all', tstat_brain)
        for i in range(nb_regressors):
            if (atlas != None): 
                res = nigsp.nifti.unfold_atlas(res[:,i], atlas)      
            p2c.io.export_nifti(res, oimg, f'{fname_out_func}_{results_labels[res]}_{emo_labels[i]}')
    
    # Export all Beta, Tstat, Lag with unfolding
    newdim_all = deepcopy(img_.header['dim'])
    newdim_all[0], newdim_all[4] = 4, int(regressors.shape[1])
    oimg_all = deepcopy(img_)
    oimg_all.header['dim'] = newdim_all   
    beta_brain_atlas = nigsp.nifti.unfold_atlas(beta_brain, atlas)      
    tstat_brain_atlas = nigsp.nifti.unfold_atlas(tstat_brain, atlas)
    lag_brain_atlas = nigsp.nifti.unfold_atlas(lag_brain, atlas) 
    
    p2c.io.export_nifti(beta_brain_atlas, oimg_all, f'{fname_out_func}_atlasunfold_beta_all')
    p2c.io.export_nifti(tstat_brain_atlas, oimg_all, f'{fname_out_func}_atlasunfold_tstat_all')
    p2c.io.export_nifti(lag_brain_atlas, oimg_all, f'{fname_out_func}_atlasunfold_lag_all')






            
def compute_lag_range (lag_max, freq):
    """
    Return the range of positive and negative lags.

    Parameters
    ----------
    fname : str or path to a text file
       Fullpath/Filename of the functional input (txt). Time should be in first dimension.
       
    Returns
    -------
    func : ndarray
        Array containing the % of change of the timeseries, with time at the last dimension.
    func_avg : ndarray
        Array containing the spatial average of the function.

    """
    nrep = np.round(lag_max * freq) * 2 + 1  # nb of regressions = nb of delays tested = n of steps 
    step = 1/freq
    lag_range = list(range(0, nrep-1, step))
    return lag_range, nrep




##########################################################################################################################

# Define global variables
TR = 1.3 # second/1.3 = TR
FREQ = 1/TR  # frequence of response variable 
FREQ_UP = 2*FREQ # freq_up en sec = 2*freqs
LAG_MAX = 4  # max lag tested in s = 4 sec
LAG_STEP = TR/2  # time between each lag tested is TR/2 seconds  
R2MODEL = 'adj_full'  # adjusted squared for multiple regressors

# Load response/dependant variable (here FMRI data)
fname_func = "./Func/t_TC_400_sub-S30_ses-3_Sintel.csv"  #func has shape 536 (time), 414 (voxels)
func = load_func(fname_func)[0]
func_avg = load_func(fname_func)[1]

# Load atlas
fname_atlas = "Schaefer2018_400Parcels_7Networks_order_FSLMNI152_1mm.nii.gz" # change with the name of the atlas
atlas, dmask_, img_ = p2c.io.load_nifti_get_mask(fname_atlas) 

# Define directory for file exportation
fname_out_func, _ = p2c.io.check_ext('.csv', os.path.basename(fname_func), remove=True)
fname_out_func = os.path.join('./S30_Sintel_meth2', fname_out_func)

func_len = func.shape[1]

# Load regressors/independent variables
# Load emotions ratings
data_emo = np.genfromtxt("./Emotions_rating/Sintel13.csv")
pad_indexes_emo = int(func_len-data_emo.shape[0])
print(f'Difference in size between the movies length and the functionnal data {pad_indexes_emo}')
onset = 72 
data_emo = np.pad(data_emo, [(onset, pad_indexes_emo-onset), (0, 0)], mode='constant', constant_values=0)  
# Load motion regressors
motions = np.genfromtxt("./Regressors/regressors_sub-S30_ses-3_Sintel.txt")
motions = motions[:, 2:]  # Remove CSF and WM for the moment
# Load physiological regressors: card and resp for now
freq_physio = 40 # freq actuelle = 40Hz
rv = np.genfromtxt("./Physio/rv_sub-30_ses-3_01")[:,1] 
hbi = np.genfromtxt("./Physio/hbi_sub-30_ses-3_01")[:,1]
rv = p2c.signal.resample_signal(rv, freq_physio, FREQ) 
hbi = p2c.signal.resample_signal(hbi, freq_physio, FREQ)
# Put every covariates to the same length
crop_idx_rv = abs(rv.shape[0] - func_len)
crop_idx_inf_rv = int(crop_idx_rv/2)
rv = rv[crop_idx_inf_rv : -(crop_idx_rv - crop_idx_inf_rv)] [..., np.newaxis] 
crop_idx_hbi = abs(hbi.shape[0] - func_len)
crop_idx_inf_hbi = int(crop_idx_hbi/2)
hbi = hbi[crop_idx_inf_hbi : -(crop_idx_hbi - crop_idx_inf_hbi)][..., np.newaxis]
pad_indexes_2 = int(motions.shape[0]-func_len)
motions = motions[int(pad_indexes_2/2): -(pad_indexes_2-int(pad_indexes_2/2))]
# Concatenates regressors
data_physio = np.append(rv, hbi, axis=1)
data_emo = np.append(data_physio, data_emo, axis=1)
# Creates labels file
emo_labels = np.append("rv", "hbi")
with open('labels.csv', 'r') as source:
    reader = csv.reader(source)
    for line in reader:
        emo_labels = np.append(emo_labels, line)

# Mean subtraction of the regressors
data_emo = data_emo - data_emo.mean(axis=-1)[..., np.newaxis]

# Upsample the HRF and the regressors before convolution
hrf = p2c.signal.create_hrf(FREQ_UP)
data_emo_up = p2c.signal.resample_signal(data_emo, 1/TR, FREQ_UP)
print("data_emo_up ", data_emo_up.shape)

# Convolve regressors with HRF 
regressors_up = np.empty_like(data_emo_up, dtype=float)
columns = data_emo_up.shape[1]  # nb of emotions
for i in range(columns):
    regr = data_emo_up[:, i]
    regressors_up[:, i] = np.convolve(regr, hrf)[:-hrf.size+1]
    
# Mean subtraction of the regressors
print("regressors_up mean ", regressors_up.mean(axis=-1)) #si = 0 on enleve la ligne
regressors_up = regressors_up - regressors_up.mean(axis=-1)[..., np.newaxis]

# Downsampled matrix of emotion regressors to recover the initial number of timepoints (= size of the functional data) for the regression
regressors = p2c.signal.resample_signal(regressors_up, FREQ_UP, 1/TR)

# Prepare empty matrices to store the lagged regressors for each brain regions
# 1st dimension = regions, 2nd = regressors
iteration = 0
rsquare_brain = list()
d_rsquare_brain = list()
beta_brain = np.empty(list(func.shape[:-1]) + [regressors.shape[-1]], 
                      dtype='float32')
tstat_brain = np.empty(list(func.shape[:-1]) + [regressors.shape[-1]], 
                       dtype='float32')
lag_brain = np.empty(list(func.shape[:-1]) + [regressors.shape[-1]], 
                     dtype='float32')

# Find lag range 
lag_range, nrep = compute_lag_range(LAG_MAX, FREQ)
step = FREQ_UP * LAG_STEP

for region in range(func.shape[0]): 
    print(f'region {region}')
    
    previous_rsquare = -1
    delta_rsquare = 0 #random initialisation of dr^2
    list_rsquare = list()
    list_d_rsquare = list()
    
    func_reg = func[region,:][..., np.newaxis]
    func_reg = func_reg.T
    
    mask = np.ones(func_reg.shape[:-1])
    mask = mask.astype('bool')
    
    beta_region = np.empty([regressors.shape[-1]], dtype='float32')
    tstat_region = np.empty([regressors.shape[-1]], dtype='float32')
    lag_region = np.empty([regressors.shape[-1]], dtype='float32')

        
    for iteration in range(5000):
        print(f'iteration {iteration}')
        for n_reg in range(regressors.shape[1]):     

            reg = regressors_up[:, n_reg]

            # Get lag regressors
            # Create shifted version of the emotion regressors (rating convolved with hrf)
            # regr_shifts are already at the downsampled size  
            outname = '.'  # path to the file of regressors
            regr, regr_shifts = p2c.stats.get_regr(func_avg, reg, TR, FREQ_UP,
                                                   outname, lag_max=LAG_MAX,  
                                                   ext='.1D', lagged_regression=True, skip_xcorr=True) 


            # Create confouning factors matrix
            mat_conf = np.hstack((regressors[:, :n_reg], regressors[:, n_reg+1:]))
            mat_conf = np.hstack((mat_conf, motions))

            # Prepare empty matrices 
            r_square_all = np.empty([len(lag_range)], dtype='float32')
            beta_all = np.empty([len(lag_range)], dtype='float32')
            tstat_all = np.empty([len(lag_range)], dtype='float32')  

            # Compute regression
            # r2 =  % of variance in the fmri data explained by the regressor
            # aim = find for each emotion which delay has the best r2 
            if R2MODEL not in p2c.stats.R2MODEL:
                raise ValueError(f'R^2 model {R2MODEL} not supported. Supported models '
                                 f'are {p2c.stats.R2MODEL}')

            for n, i in enumerate(lag_range): # n : index of iteration  change to: in range
                regr_shifts_n = regr_shifts[:, n]   # TR vs each delay. regr_shifts = delayed versions

                (beta_all[n],
                 tstat_all[n],
                 r_square_all[n]) = p2c.stats.regression(func_reg,
                                                         regr_shifts_n,
                                                         mat_conf,
                                                         r2model=R2MODEL)



            # Find lag for the highest rsquare
            lag_idx = np.argmax(r_square_all, axis=-1)
            lag = (lag_idx * step) / FREQ_UP - LAG_MAX 
            
                         
            beta_region[n_reg] = beta_all[lag_idx]
            tstat_region[n_reg] = tstat_all[lag_idx] 
            lag_region[n_reg] = lag

            lagged_reg = np.squeeze(regr_shifts[:,lag_idx])
            regressors[:, n_reg] = lagged_reg
            regressors_up[:, n_reg] = p2c.signal.resample_signal(lagged_reg, 1/TR, FREQ_UP)

            # Compute delta rsquare to see if the zscore is padding
            # we want each voxel to have less than DZ as difference in rsquare from one step to another
            r_square = r_square_all[lag_idx]
            delta_rsquare = abs(r_square - previous_rsquare)
            list_d_rsquare.append(delta_rsquare)
            list_rsquare.append(r_square)
            previous_rsquare = r_square
            
            
            if delta_rsquare <= 0.01:
                #print(f'Delta R2 reached the threshold: dR2 = {delta_rsquare}')

                if iteration > 1 or n_reg == (regressors.shape[1]-1):
                    # Save tables  with all r square, t and beta values 
                    beta_brain[region,:] = beta_region
                    tstat_brain[region,:] = tstat_region
                    lag_brain[region,:] = lag_region
                    print("break")
                    rsquare_brain.append(list_rsquare)
                    d_rsquare_brain.append(list_d_rsquare)
                    break
                else:
                    continue
            else:
                continue
               
        break 


results = [beta_brain, tstat_brain, lag_brain]
results_labels = ['beta', 'tstat', 'lag']
export_results(results, results_labels, atlas=atlas)


# Export dR2 and R2
rsquare_brain = np.array(rsquare_brain)
d_rsquare_brain = np.array(d_rsquare_brain)
np.savetxt(fname_out_func+'rsquare_brain', rsquare_brain)
np.savetxt(fname_out_func+'d_rsquare_brain', d_rsquare_brain)



