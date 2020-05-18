#%% IMPORT LIBRARY
import numpy as np
import math
import sim
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import os

directory = 'fig'
try:
    os.stat(directory)
except:
    os.mkdir(directory)

############## PARAMETERS #####################
#datasets = ['dna','letter','pendigits','usps','mushrooms'] #'dna','letter','pendigits','usps','mushrooms'
datasets = ['thermostat']
runs = 1
noise_type = ['constant'] #['constant'] see gennoise() in noise.py for valid options
noise_steps = [1000]
clean_batches = 1 # first is init batch
#model_types = ['rf'] #'knn', 'nearestcentroid', 'svm', 'gaussianprocess', 'rf'
model_type = 'mlp' # see getmodel() in sim.py for valid options
quality_model_type = 'knn'# 'knn','nearestcentroid', 'svm', 'gaussianprocess', 'rf'] # see getmodel() in sim.py for valid options

quality_model_selection_type = 'simple'

sampler_type = 'none'#['randomunder', 'allknn', 'condensednn', 'editednn', 'repeatededitednn', 'tomeklinks', 'randomover','quality'] # see getsampler() in sim.py for valid options ## good ones:  'randomunder', 'allknn', 'condensednn', 'editednn', 'repeatededitednn', 'tomeklinks', 'randomover',
quality_sampler_type =  'none'#['nearmiss', 'randomover']#['randomunder', 'nearmiss', 'allknn', 'condensednn', 'editednn', 'repeatededitednn', 'tomeklinks', 'randomover', 'smote', 'adasyn', 'smotenc', 'quality'] # see getsampler() in sim.py for valid options

loops=len(datasets) ## to update
n_init = 6000
n_batch = 300
n_samples = 57000

time_horizon = np.math.floor((n_samples - n_init) / n_batch)
accuracy_sel = np.zeros((time_horizon, runs))
accuracy_all = np.zeros((time_horizon, runs))
accuracy_init = np.zeros(runs)
accuracy_all_clean = np.zeros((time_horizon, runs))
improved_sample_last = np.zeros((time_horizon, loops))
improved_sample_init = np.zeros((time_horizon, loops))
acc_imp = np.zeros((2, loops))
accuracy_imp = np.zeros((time_horizon, loops))
improved_sample_init_th = np.zeros(loops)
improved_sample_last_th = np.zeros(loops)
acc_imp_CL = np.zeros(loops)
acc_imp_Sel = np.zeros(loops)
acc_imp_Imp = np.zeros(loops)
result_tab=[]
result_tab2=[]
clean = np.zeros((time_horizon, runs))
selected = np.zeros((time_horizon, runs))
overlap = np.zeros((time_horizon, runs))
selected_sampled = np.zeros((time_horizon, runs))

for j,dataset in enumerate(datasets):
#for j,sampler_type in enumerate(sampler_types):
    if dataset=='dna':
        noise_min = [0.4]
        noise_max =[0.2]
    elif dataset=='letter':
        noise_min = [0.8]
        noise_max =[0.2]
    elif dataset=='pendigits':
        noise_min = [0.8]
        noise_max =[0.2]
    elif dataset=='usps':
        noise_min = [0.8]
        noise_max =[0.2]
    elif dataset=='mushrooms':
        noise_min = [0.4]
        noise_max =[0.2]
    elif dataset=='thermostat':
        noise_min = [0.3]
        noise_max =[0.3]
    elif dataset=='tasks-quarter':
        noise_min = [0.3]
        noise_max =[0.3]
    else:
        print("unknown dataset ",dataset)

    for i in range(runs):
        s = sim.Sim(dataset, n_init, n_batch, n_samples, noise_min, noise_max, noise_type, noise_steps, model_type, quality_model_type, clean_batches, sampler_type, quality_sampler_type, quality_model_selection_type)
        accuracy_init[i], accuracy_sel[:,i], accuracy_all[:,i], accuracy_all_clean[:,i], clean[:,i], selected[:,i], overlap[:,i], selected_sampled[:,i] = s.run_sim()

    accuracy_sel_mean = np.mean(accuracy_sel, axis=1)
    accuracy_sel_std = np.std(accuracy_sel, axis=1)
    accuracy_all_mean = np.mean(accuracy_all, axis=1)
    accuracy_all_std = np.std(accuracy_all, axis=1)
    accuracy_all_clean_mean = np.mean(accuracy_all_clean, axis=1)
    accuracy_all_clean_std = np.std(accuracy_all_clean, axis=1)
    accuracy_init_mean = np.mean(accuracy_init)
    accuracy_init_std = np.std(accuracy_init)

    clean_mean = np.mean(clean, axis=1)
    clean_std = np.std(clean, axis=1)
    selected_mean = np.mean(selected, axis=1)
    overlap_mean = np.mean(overlap, axis=1)
    selected_std = np.std(selected, axis=1)
    overlap_std = np.std(overlap, axis=1)
    selected_sampled_mean = np.mean(selected_sampled, axis=1)
    selected_sampled_std = np.std(selected_sampled, axis=1)

    th_ix = math.ceil((1000 - n_init)/n_batch)
    all_acc_last = accuracy_all_mean[th_ix]
    seltrue_acc_last = accuracy_all_clean_mean[th_ix]
    all_acc_init = accuracy_all_mean[0]
    sel_acc_init = accuracy_sel_mean[0]
    all_err_init = 1-accuracy_all_mean[0]
    all_err_last = 1-accuracy_all_mean[th_ix]

    acc_imp_CL[j] = (accuracy_sel_mean[th_ix] - sel_acc_init) / sel_acc_init
    acc_imp_Sel[j] = (accuracy_sel_mean[th_ix] - all_acc_last) / all_acc_last
    acc_imp_Imp[j] = (accuracy_sel_mean[th_ix] - seltrue_acc_last) / seltrue_acc_last

    improved_sample_last[:,j] = ((1-accuracy_sel_mean) - all_err_last) / all_err_last
    improved_sample_init[:,j] = ((1-accuracy_sel_mean) - all_err_init) / all_err_init
    improved_sample_init_th[j] = improved_sample_init[th_ix,j]
    improved_sample_last_th[j] = improved_sample_last[th_ix,j]

    flag05=0
    flag15=0
    for batch_ix in np.arange(time_horizon):
        accuracy_imp[batch_ix,j] = ((accuracy_sel_mean[batch_ix]) - sel_acc_init) / sel_acc_init
        if accuracy_imp[batch_ix,j] > 0.05 and flag05==0:
            flag05=batch_ix
            acc_imp[0,j] = n_batch*batch_ix
        if accuracy_imp[batch_ix,j] > 0.15 and flag15==0:
            flag15=batch_ix
            acc_imp[1,j] = n_batch*batch_ix

    result_tab2.append([dataset, sel_acc_init*100, accuracy_sel_std[0]*100, all_acc_last*100, accuracy_all_std[th_ix]*100, accuracy_sel_mean[th_ix]*100, accuracy_sel_std[th_ix]*100, seltrue_acc_last*100, accuracy_all_clean_std[th_ix]*100, int(acc_imp_CL[j]*1000)/10, int(acc_imp_Sel[j]*1000)/10, int(-acc_imp_Imp[j]*1000)/10,100*sum(selected_mean)/(n_batch*time_horizon)])

    #%% plots & table
    n = len(accuracy_sel)
    x = np.arange(n)

    filebase = '%s/%s_%s_%f_%f_%s_%s_%d_%d' % (directory, dataset, noise_type[0], noise_min[0], noise_max[0], model_type, quality_model_type, n_init, n_batch)

    # save to disk
    df_error = pd.DataFrame(data={'data_batch': x, 'accuracy_sel_mean': accuracy_sel_mean, 'accuracy_sel_std': accuracy_sel_std, 'accuracy_all_mean': accuracy_all_mean, 'accuracy_all_std': accuracy_all_std, 'accuracy_all_clean_mean': accuracy_all_clean_mean, 'accuracy_all_clean_std': accuracy_all_clean_std})
    df_error.to_csv(filebase + "_error.csv")


    fig_error = plt.figure()
    fig_error = plt.errorbar(x, accuracy_sel_mean, yerr=accuracy_sel_std, fmt='ko-', label='Duo')
    fig_error = plt.errorbar(x, accuracy_all_mean, yerr=accuracy_all_std, fmt='y+',linestyle='dashed', label='No-Sel')
    #fig_error = plt.errorbar(x, accuracy_init_mean*np.ones(len(x)), fmt=':b',linewidth=3, label='Init')#yerr=accuracy_init_std*np.ones(len(x)),
    fig_error = plt.errorbar(x, accuracy_all_clean_mean, yerr=accuracy_all_clean_std, fmt='d-.m',label='Opt-Sel')
    plt.ylabel('Classification Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower left')
    #plt.axis([0, 17, 0.64, 0.86])
    #plt.title('Gain/loss over all: %.1f ' % (100.0*improved_sample_last[th_ix,j]))
    plt.grid(True)
    #plt.savefig('figures/NIPS_dynamic_usps_100runs.eps')

    fig_noise = plt.figure()

    df_noise = pd.DataFrame(data={'data_batch': x, 'clean': clean[:,0], 'selected': selected[:,0], 'overlap': overlap[:,0]})

    df_noise.to_csv(filebase + "_noise.csv")

#    fig_noise = plt.errorbar(x, clean_mean, yerr=clean_std, fmt='--m', label='Clean')
#    fig_noise = plt.errorbar(x, selected_mean, yerr=selected_std, fmt='ko-', label='Selected')
#    fig_noise = plt.errorbar(x, overlap_mean, yerr=overlap_std, fmt=':xc', label='Selected & Clean')
#    #fig_noise = plt.errorbar(x, selected_sampled_mean, yerr=selected_sampled_std, fmt='gd-.', label='Selected Sampled')
    fig_noise = plt.errorbar(x, clean[:,0],  fmt='--m', label='Clean')
    fig_noise = plt.errorbar(x, selected[:,0], fmt='ko-', label='Selected')
    fig_noise = plt.errorbar(x, overlap[:,0], fmt=':xc', label='Selected & Clean')
    plt.ylabel('#-samples')
    plt.legend(loc='lower center')
    plt.xlabel('Epochs')
    #plt.axis([0, 17, 0, 50])
    plt.grid(True)
    #plt.savefig('figures/NIPS_dynamic_usps_100runs_samples.eps')
    plt.show()

    print(tabulate(result_tab2, headers=["Dataset", "Init","+-","No-Sel","+-","Sel","+-","Opt-Sel","+-","CL","Selection","Dist to opt","% used samp."]))
