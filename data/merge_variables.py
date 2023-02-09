"""
"""
import uproot
import os
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp as KS


current_path = os.path.dirname(__file__)

tree = 't_M0pipi;2'
filepath = '../root_files'
vars = ['M0_p', 'M0_MKK']
filenames = ['tree_B0PiPi_mc.root', 'tree_B0sKK_mc.root']

files = [os.path.join(
    current_path, filepath, filename) for filename in filenames]

[tree_pi, tree_k] = [uproot.open(file)[tree] for file in files]


def mix_function(y, x, m):
    return y + m*x


[v1_pi, v2_pi] = [tree_pi[var].array(library='np') for var in vars]
[v1_K, v2_K] = [tree_k[var].array(library='np') for var in vars]


v1lim = [min(min(v1_pi), min(v1_K)), max(max(v1_pi), max(v1_K))]
v2lim = [min(min(v2_pi), min(v2_K)), max(max(v2_pi), max(v2_K))]

v1_pi = (v1_pi - v1lim[0])/(v1lim[1]-v1lim[0])
v2_pi = (v2_pi - v2lim[0])/(v2lim[1]-v2lim[0])
v1_K = (v1_K - v1lim[0])/(v1lim[1]-v1lim[0])
v2_K = (v2_K - v2lim[0])/(v2lim[1]-v2lim[0])

n_try = 1000

pars = np.linspace(-100., 100., n_try)
ks_stats = np.zeros(n_try)
idx_sel = 0
max_stat = 0.

for i in range(len(pars)):
    v_merge_pi = mix_function(v1_pi, v2_pi, pars[i])
    v_merge_K = mix_function(v1_K, v2_K, pars[i])
    ks_stats[i] = KS(v_merge_pi, v_merge_K, alternative='twosided').statistic
    if i != 0:
        if ks_stats[i] > max_stat:
            max_stat = ks_stats[i]
            idx_sel = i

v_merge_pi = mix_function(v1_pi, v2_pi, pars[idx_sel])
v_merge_K = mix_function(v1_K, v2_K, pars[idx_sel])
print(pars[idx_sel], max_stat)
print(KS(v1_pi, v1_K, alternative='twosided').statistic)
print(KS(v2_pi, v2_K, alternative='twosided').statistic)

plt.figure(1)
plt.subplot(2, 1, 1)
plt.hist(v_merge_pi, 100, color='blue', histtype='step')
plt.hist(v_merge_K, 100, color='red', histtype='step')
plt.subplot(2, 1, 2)
plt.hist(v1_pi, 100, color='blue', histtype='step')
plt.hist(v1_K, 100, color='red', histtype='step')
plt.hist(v2_pi, 100, color='blue', histtype='step')
plt.hist(v2_K, 100, color='red', histtype='step')
plt.show()
