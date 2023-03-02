"""
"""
import ROOT
import uproot
import os
import numpy as np
import time 

t1 = time.time()
current_path = os.path.dirname(__file__)
rel_path = 'root_files'
filenames_in = ['toyMC_B0PiPi.root', 'toyMC_B0sKK.root']
filenames_out = ['B0PiPi_MC.root', 'B0sKK_MC.root','Bhh_data.root']
tree = 't_M0pipi;1'

filepaths_in = [os.path.join(current_path, rel_path, file)
             for file in filenames_in]
dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths_in]

filepaths_out = [os.path.join(current_path, rel_path, file)
             for file in filenames_out]

n_evts_toymc_pi = dataframes[0].Count().GetValue()
n_evts_toymc_k = dataframes[1].Count().GetValue()

num_MC = 250000  # Number of generated MC events
num_data = 50000  # Number of generated data events

f = 0.42  # Ideal fraction of Kaons in generated data file

num_species = [int((1-f)*num_data), int(f*num_data)]
num_pions, num_kaons = num_species

print(f'Actual fraction of Kaons = {num_kaons/num_data}')

if (n_evts_toymc_pi < num_MC + num_pions) or (n_evts_toymc_k < num_MC + num_kaons):
    print("ERROREE")
    pass

vars = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
        'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2', 'h2_thetaC0',
        'h2_thetaC1', 'h2_thetaC2', 'flag')

dataframes[0] = dataframes[0].Define("flag", "0")
dataframes[1] = dataframes[1].Define("flag", "1")

df_mc_pi = dataframes[0].Range(num_MC) # takes the first num_MC events of the input toyMC files
df_mc_pi.Snapshot("tree", filepaths_out[0], vars) # creates a .root file with the chosen vars as branches

df_mc_k = dataframes[1].Range(num_MC)
df_mc_k.Snapshot("tree", filepaths_out[1], vars)


df_data_pi = dataframes[0].Range(num_MC, num_MC+num_species[0]) # takes the rest of the input toyMC to be used as data
# tmp_df_data_pi.Snapshot("tree_data", filenames_out[2], vars)

df_data_k = dataframes[1].Range(num_MC, num_MC+num_species[1])
# tmp_df_data_k.Snapshot("tree_data", filenames_out[3], vars)


var_list = []

file = uproot.recreate(filepaths_out[2])

for idx in range(len(vars)):
    v_temp_pi = df_data_pi.AsNumpy()[vars[idx]]
    v_temp_k = df_data_k.AsNumpy()[vars[idx]]
    v_temp = np.concatenate((v_temp_pi, v_temp_k), axis=0)
    var_list.append(v_temp)

var_array = np.stack(var_list, axis=1)
print(np.shape(var_array))

np.random.shuffle(var_array)

var_dictionary = {}
for idx in range(len(vars)-1):
    var_dictionary.update({vars[idx]: var_array[:, idx]}) # !!! Could be made more DRY since it's the same as AsNumpy() above ?

file["tree"] = var_dictionary

'''
plt.figure(1)
x = np.linspace(1, 10000, 10000)
plt.plot(x, var_array[:, -1])
plt.savefig("prova.png")
'''
file["tree"].show()
file["tree"].close()

t2 = time.time()

print(f'Time elapsed = {t2-t1} s')
