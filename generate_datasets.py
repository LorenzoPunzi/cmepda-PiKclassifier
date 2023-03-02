"""
"""
import ROOT
import uproot
import os
from data.import_functions import loadvars
import numpy as np
import matplotlib.pyplot as plt


current_path = os.path.dirname(__file__)
rel_path = 'root_files'
filenames_in = ['toyMC_B0PiPi.root', 'toyMC_B0sKK.root']
filenames_out = ['tree_B0PiPi_MC.root', 'tree_B0sKK_MC.root',
                 'tree_Pi_MC.root', 'tree_K_MC.root', 'tree_Bhh_data.root']
tree = 't_M0pipi;1'

filepaths = [os.path.join(current_path, rel_path, file)
             for file in filenames_in]
dataframes = [ROOT.RDataFrame(tree, filepath) for filepath in filepaths]

n_evts_mc_pi = dataframes[0].Count().GetValue()
n_evts_mc_k = dataframes[1].Count().GetValue()

num_MC = 100000  # Number of events in each MC
num_data = 10000  # Number of data events

f = 0.42  # Fraction of Kaons in data file
num_species = [int((1-f)*num_data), int(f*num_data)]
num_pions, num_kaons = num_species
# num_kaons = int(f*num_data)
# num_pions = num_data - num_kaons
print(num_data)
print(num_pions+num_kaons)

if (n_evts_mc_pi < num_MC + num_pions) or (n_evts_mc_k < num_MC + num_kaons):
    print("ERROREE")
    pass

vars = ('M0_Mpipi', 'M0_MKK', 'M0_MKpi', 'M0_MpiK', 'M0_p', 'M0_pt',
        'M0_eta', 'h1_thetaC0', 'h1_thetaC1', 'h1_thetaC2', 'h2_thetaC0',
        'h2_thetaC1', 'h2_thetaC2', 'flag')

dataframes[0] = dataframes[0].Define("flag", "0")
dataframes[1] = dataframes[1].Define("flag", "1")

tmp_df_mc_pi = dataframes[0].Range(num_MC)
tmp_df_mc_pi.Snapshot("tree_mc", filenames_out[0], vars)

tmp_df_mc_k = dataframes[1].Range(num_MC)
tmp_df_mc_k.Snapshot("tree_mc", filenames_out[1], vars)


tmp_df_data_pi = dataframes[0].Range(num_MC, num_MC+num_species[0])
# tmp_df_data_pi.Snapshot("tree_data", filenames_out[2], vars)

tmp_df_data_k = dataframes[1].Range(num_MC, num_MC+num_species[1])
# tmp_df_data_k.Snapshot("tree_data", filenames_out[3], vars)


var_list = []

file = uproot.recreate("tree_data_prova.root")

for idx in range(len(vars)):
    v_temp_pi = tmp_df_data_pi.AsNumpy()[vars[idx]]
    v_temp_k = tmp_df_data_k.AsNumpy()[vars[idx]]
    v_temp = np.concatenate((v_temp_pi, v_temp_k), axis=0)
    var_list.append(v_temp)

var_array = np.stack(var_list, axis=1)
print(np.shape(var_array))

np.random.shuffle(var_array)

var_dictionary = {}
for idx in range(len(vars)):
    var_dictionary.update({vars[idx]: var_array[:, idx]})

file["tree"] = var_dictionary

'''
plt.figure(1)
x = np.linspace(1, 10000, 10000)
plt.plot(x, var_array[:, -1])
plt.savefig("prova.png")
'''
file["tree"].show()
file["tree"].close()


'''
v1 = dataframes[0].Range(num_MC).AsNumpy()["flag"]
print(v1)
v2 = dataframes[1].Range(num_MC).AsNumpy()["flag"]
print(v2)

vec = np.concatenate((v1, v2), axis=0)
vec = vec[99990:100010]
print(type(vec))
print(vec)


# v1, v2 = loadvars(filenames_out[2], filenames_out[3],
#                   "tree_data", vars, flag_column=False)


v = np.concatenate((v1, v2), axis=0)
np.random.shuffle(v)


'''

'''


t.Fill()
t.Write()
f.Close()




for idx in range(len(vars)):
    var_dictionary.update({vars[idx]: v[:, idx]})

v_prova = dataframes[0].Range(num_MC, num_MC+num_species[0]).AsNumpy()




'''
