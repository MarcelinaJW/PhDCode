#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

#%%
import pandas as pd
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
# import seaborn as sns
# import sys
import scipy
from glob import glob
import pySuStaIn
import os 




data_dir='/Users/mw4217/Desktop/Marcelina/SuStaIn'

#%% Load data in 

df = pd.ExcelFile(os.path.join(data_dir,'Sustain_Oligo.xlsx')).parse('all',header=0)


region = ['Frontal', 'BasalGanglia', 'Hippocampus', 'Cerebellum',
       'Midbrain', 'Pons', 'Medulla', 'SpinalCord' ]
df[region].head()

# convert to numeric
for col in region:
    df.loc[:,col] = pd.to_numeric(df[col].values,
                                      errors='coerce')
    
# how many patients and controls do we have

df['Category'] = df['Category']='PD'

select_patient = df['Category']=='PD'
print(df.loc[select_patient,region].shape)


# Set up data for new version of SuStaIn
#The objective here is two create to variables:

#* p_NL -- a subject x region matrix indicating the p that a region is normal for a subject.
#* p_score -- a subject x (region * score) matrix indicating p that a region is a given score.

#Ps should be non-zero. 
#%% Do pre-amble stats
# Extract patients into unique variable 
patientdata = np.array(df.loc[select_patient,region],copy=True)

# Output the number of subjects and region
N_Subjects = patientdata.shape[0]
N_region = patientdata.shape[1]

# Output model scores (it doesn't consider 0s?)
# ModelScores = np.array([1,2,3,4,5,6])
ModelScores = np.array([1,2,3,4])
N_ModelScores = ModelScores.shape[0]

# Output the datascores 
#DataScores = np.array([0,1,2,3,4,5,6])
DataScores = np.array([0,1,2,3,4])
N_DataScores = DataScores.shape[0]

# Create labels for SuStaIn
SuStaInLabels = region

#Set sigma
#  can be changed i.e smaller values higher prob of being the category attributed
sigma = 0.5

#Create a probability distribution whereby values next to eachother have higher likelihood of being conflated
#   Needs to be NxN where N=number of unique score values (i.e., DataScores)
# probability of the category being the value attributed
prob_dist = np.array([scipy.stats.norm.pdf(DataScores,loc=0,scale=sigma),
                      scipy.stats.norm.pdf(DataScores,loc=1,scale=sigma),
                       scipy.stats.norm.pdf(DataScores,loc=2,scale=sigma),
                       scipy.stats.norm.pdf(DataScores,loc=3,scale=sigma),
                       scipy.stats.norm.pdf(DataScores,loc=4,scale=sigma),
                       scipy.stats.norm.pdf(DataScores,loc=5,scale=sigma),
                        scipy.stats.norm.pdf(DataScores,loc=6,scale=sigma)])

# Divide the probability distribution by the sum of probability for each region
# to make the probdist equal 100%
prob_dist_norm = prob_dist/np.matlib.repmat(np.sum(prob_dist,axis=0),7,1)

# Set the minimum to 0.01 and divide by the sum
# small chance of being another value i.e being 3 when it is 1, but not setting the chance to 0
prob_dist_norm = np.maximum(prob_dist_norm,0.01)/np.matlib.repmat(np.sum(np.maximum(prob_dist_norm,0.01),axis=0),7,1)

# Get the distribution
p_nl_dist = prob_dist_norm[0,]
#  prob of not being 0
p_score_dist = prob_dist_norm[1:,]

# Output a N*B array with the probabilty distributions that value is normal 
# prob of a region being normal (no path)
prob_nl = np.ones((N_Subjects,N_region))/(N_ModelScores+1)
for index in range(N_DataScores):
    prob_nl[patientdata==DataScores[index]] = p_nl_dist[index]
 
# Output an N*B*S matrix with the probability that a region is that region given the score (aka 1 and 2 will have closer distributions than 1 and 3...)
# gaussian distibution
prob_score = np.ones((N_Subjects,N_region,N_ModelScores))/(N_ModelScores+1)
for n in range(N_region):
    for index in range(N_DataScores):
        for score in range(N_ModelScores):
            prob_score[patientdata[:,n]==DataScores[index],n,score] = p_score_dist[score,index]
            
#%% Saving Data 
score_vals = np.matlib.repmat(ModelScores,N_region,1)

os.makedirs(os.path.join(data_dir, 'input'))

# saving as npy
np.save(os.path.join(data_dir,'input/score_vals_gaussian'),score_vals)
np.save(os.path.join(data_dir,'input/SuStaInLabels_gaussian'),SuStaInLabels)
np.save(os.path.join(data_dir,'input/prob_nl_gaussian'),prob_nl)
np.save(os.path.join(data_dir,'input/prob_score_gaussian'),prob_score)

# saving panda as excel
with pd.ExcelWriter(os.path.join(data_dir,'input/data_gaussian.xlsx')) as writer:
    pd.DataFrame(patientdata,columns=SuStaInLabels).to_excel(writer,sheet_name="score")
    pd.DataFrame(prob_nl,columns=SuStaInLabels).to_excel(writer,sheet_name="p_0")
    pd.DataFrame(prob_score[:,:,0],columns=SuStaInLabels).to_excel(writer,sheet_name="p_1")
    pd.DataFrame(prob_score[:,:,1],columns=SuStaInLabels).to_excel(writer,sheet_name="p_2")
    pd.DataFrame(prob_score[:,:,2],columns=SuStaInLabels).to_excel(writer,sheet_name="p_3")
    pd.DataFrame(prob_score[:,:,3],columns=SuStaInLabels).to_excel(writer,sheet_name="p_4")
    # pd.DataFrame(prob_score[:,:,4],columns=SuStaInLabels).to_excel(writer,sheet_name="p_5")
    #pd.DataFrame(prob_score[:,:,5],columns=SuStaInLabels).to_excel(writer,sheet_name="p_6")

    
biomarker_index = np.arange(len(SuStaInLabels))
columns = ['score','p_0','p_1','p_2','p_3', 'p_4']
with pd.ExcelWriter(os.path.join(data_dir,'input/probabilities_gaussian.xlsx')) as writer:
    for index in biomarker_index:
        temp_array = np.zeros((N_Subjects,len(columns)))
        temp_array[:,0] = patientdata[:,index]
        temp_array[:,1] = prob_nl[:,index]
        temp_array[:,2] = prob_score[:,index,0]
        temp_array[:,3] = prob_score[:,index,1]
        temp_array[:,4] = prob_score[:,index,2]
        temp_array[:,5] = prob_score[:,index,3]
        #temp_array[:,6] = prob_score[:,index,4]
   
        pd.DataFrame(temp_array,columns=columns).to_excel(writer,sheet_name=SuStaInLabels[index])
        #i added temp array 5 and 6
#%%
# save np.load

# to load all the data
np_load_old = np.load
# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

score_vals = np.load(os.path.join(data_dir,'input/score_vals_gaussian.npy'))
SuStaInLabels = np.load(os.path.join(data_dir,'input/SuStaInLabels_gaussian.npy'))
prob_nl = np.load(os.path.join(data_dir,'input/prob_nl_gaussian.npy'))
prob_score = np.load(os.path.join(data_dir,'input/prob_score_gaussian.npy'))

# restore np.load for future normal usage
np.load = np_load_old

outdir = os.path.join(data_dir,'output')

#%% 

#os.makedirs('/Users/mo920/Desktop/Maria/PhD/IHC/SuStaIn/output/pickle_files')
os.makedirs('/Users/mw4217/Desktop/Marcelina/SuStaIn/output/pickle_files')
# setting up sustain to run

#randomise sequence, but then its ordered

#nstartp - randomise 10 times (higher numbers take longer)
N_startpoints = 10

#subtypes we will identify
N_S_max = 3
N_iterations_MCMC = int(1e5)
output_folder = outdir
dataset_name = 'sample_data_gaussian'
sustain_input = pySuStaIn.OrdinalSustain(prob_nl,
                                  prob_score,
                                  score_vals,
                                  SuStaInLabels,
                                  N_startpoints,
                                  N_S_max, 
                                  N_iterations_MCMC, 
                                  output_folder, 
                                  dataset_name, 
                                  False)

[samples_sequence, samples_f, 
ml_subtype, prob_ml_subtype, ml_stage, 
prob_ml_stage, prob_subtype_stage] = sustain_input.run_sustain_algorithm(plot=True, cmap="marcelina_colours")

#%% Loading the data into sustain 
# s represents the number of splits aka, 0 split = 1 subtype, 1 split = 2 subtypes, 2 split = 3 subtypes, etc... 
s = 1 

# Loading that output data into python 
pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pd.read_pickle(pickle_filename_s)


samples_sequence = pk["samples_sequence"]
samples_f = pk["samples_f"]

M = len(df[region]) 

# use this information to plot the positional variance diagrams
tmp=pySuStaIn.OrdinalSustain._plot_sustain_model(sustain_input,samples_sequence,samples_f,M, cmap="marcelina_colours")

#%% Putting subtype and stage back into our dataframe 

#Create a dataframe of just patients 
df_patient = df[select_patient].copy()

# let's take a look at all of the things that exist in SuStaIn's output (pickle) file
pk.keys()

# The SuStaIn output has everything we need. We'll use it to populate our dataframe.
s = 1
pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'
pk = pd.read_pickle(pickle_filename_s)

for variable in ['ml_subtype', # the assigned subtype
                 'prob_ml_subtype', # the probability of the assigned subtype
                 'ml_stage', # the assigned stage 
                 'prob_ml_stage',]: # the probability of the assigned stage
    
    # add SuStaIn output to dataframe
    df_patient.loc[:,variable] = pk[variable]
#%% Loading subtype and stage, putting them back into our dataframe NEW ADDED

#Extract cases and subtypes
#Set splits This here represents the number of data splits, so 1 subtype = 0, 2 subtypes = 1, etc..

s = 1


#Create a dataframe of just patients This creates a data frame of just patients

df_patient = df[select_patient].copy()


# The SuStaIn output has everything we need. We'll use it to populate our dataframe. This loads in the sustain output, it contains all the outputs from sustain

pickle_filename_s = output_folder + '/pickle_files/' + dataset_name + '_subtype' + str(s) + '.pickle'

pk = pd.read_pickle(pickle_filename_s)


#Here we are indexing our pk file to get the subtype/stage + probabilities of those
# and assigning them to a new column in our dataframe This assigns the subtype a patient belongs to and what stage they are at in the disease to your patient data frame

df_patient.loc[:,'ml_subtype'] = pk['ml_subtype']

df_patient.loc[:,'ml_stage'] = pk['ml_stage']

df_patient.to_csv(f"{outdir}/subtypes.csv")

# %%
#CVIC to confirm number of subtypes
from sklearn.model_selection import KFold

N_folds = 10

kf = KFold(
    n_splits=N_folds,
    shuffle=True,
    random_state=42
)

test_idxs = []

for _, test_index in kf.split(prob_nl):
    test_idxs.append(test_index)

    CVIC, loglike_matrix = sustain_input.cross_validate_sustain_model(
    test_idxs
)

print("CVIC:")
print(CVIC)

import matplotlib.pyplot as plt
import numpy as np



# %%
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,4))

plt.plot(
    np.arange(1, len(CVIC)+1),
    CVIC,
    marker='o'
)

plt.xlabel("Number of subtypes")
plt.ylabel("CVIC")
plt.xticks(np.arange(1, len(CVIC)+1))
plt.tight_layout()
plt.show()
# %%
