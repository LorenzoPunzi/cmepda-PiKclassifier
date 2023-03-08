#!/usr/bin/env python
# Copyright (C) 2023 - Lorenzo Punzi (l.punzi@studenti.unipi.it),
#                      Ruben Forti (r.forti1@studenti.unipi.it)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
"""
import argparse
import time
import matplotlib.pyplot as plt
from machine_learning.deepnn import dnn
from template_fit.template_fit_var import fit_mc_template, global_fit
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from utilities.gen_from_toy import gen_from_toy
from utilities.dnn_settings import dnn_settings
from utilities.utils import default_rootpaths, default_txtpaths, default_vars,\
                            find_cut, roc, plot_rocs

# print(" ----------------------------------------------- ")
# print("|  Welcome to the PiK Classifier package!       |")
# print("|                                               |")
# print("|  Authors: Lorenzo Punzi, Ruben Forti          |")
# print("|  Release: 1.0  -  march 2023                  |")
# print(" ----------------------------------------------- ")


default_toyMC_path = ('data/root_files/toyMC_B0PiPi.root',
                      'data/root_files/toyMC_B0sKK.root')

parser = argparse.ArgumentParser(prog='PiK classifier',
                                 description='What the program does',
                                 epilog='Text at the bottom of help')

parser.add_argument('-g', '--generate', action='store_true',
                    help='Generates MC and mixed datasets from the toyMC')

parser.add_argument('-rp', '--rootpaths', nargs=2, default=default_toyMC_path,
                    help='Path of the toyMC root files taken as input')

parser.add_argument('-tr', '--tree', default='t_M0pipi;1',
                    help='Name of the tree of the toyMC root files where the variables are stored')

parser.add_argument('-rds', '--root_datasets', nargs=3, default=default_rootpaths(),
                    help='Path of the datasets generated from the toyMC')

parser.add_argument('-f', '--fraction', type=float, default=0.42,
                    help='Fraction of Kaons in the mixed dataset you want to generate')


parser.add_argument('-v', '--variables', nargs='+', default=default_vars(),
                    help='Variables you want to treat')

parser.add_argument('-vfit', '--var_fit', default='M0_Mpipi',
                    help='Variable on which template fit is performed')

parser.add_argument('-vcut', '--var_cut', default='M0_Mpipi',
                    help='Variable on which cut evaluation is performed')

parser.add_argument('-t', '--type', nargs='+', default='all',
                    choices=['tfit', 'dnn', 'dtc', 'vcut', 'all'],
                    help='Type of the analysis to be performed')

parser.add_argument('-cr', '--cornerplot', action='store_true',
                    help='Choice to generate and save the cornerplot of the variables given in input')

parser.add_argument('-fp', '--figpath',
                    help='Path where figures are going to be saved')

args = parser.parse_args()


filepaths = args.root_datasets
figpath = args.figpath
tree = args.tree

if args.generate:
    # Generates the datasets with the requested fraction of Kaons in
    # the mixed sample. If the following two quantities are BOTH set to
    # zero, the function generates the datasets with the maximum
    # possible number of events
    NUM_MC = 0
    NUM_DATA = 0
    gen_from_toy(filepaths_in=tuple(args.rootpaths), tree=tree, f=args.fraction,
                 n_mc=NUM_MC, n_data=NUM_DATA, vars=tuple(args.variables))


# Initialize the appropriate analysis-methods list, also removing duplicates
if 'all' in args.type:
    analysis = ['all']
else:
    analysis = []
    a = [analysis.append(item) for item in args.type if item not in analysis]


for opt in analysis:

    if opt in ["tfit", "all"]:
        # ~~~~~~~~ Setup of the template fit - free to edit ~~~~~~~~~~~
        Nbins_histo = 1000
        histo_lims = (5.0, 5.6)  # Limits of the histograms
        fit_range = (5.02, 5.42)  # Range where the functions are fitted
        p0_pi = (1e5, 0.14, 5.28, 0.08, 5.29, 0.02)
        p0_k = (1e5, 0.96, 1.7, 0.6, 5.29, 0.7, 5.18, 0.05)
        figures = True
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        t0 = time.time()
        print("\nTemplate fit with ROOT - working...\n")
        var = args.var_fit

        templ_pars_pi = fit_mc_template(
            filepaths[0], 'tree;1', var, DoubleGaussian(fit_range, pars=p0_pi),
            Nbins=Nbins_histo, histo_lims=histo_lims,
            histo_title=f'{var} distribution (B0->PiPi MC)', savefig=figures,
            img_name='fig/template_fit_pi.png')
        templ_pars_k = fit_mc_template(
            filepaths[1], 'tree;1', var, GaussJohnson(fit_range, pars=p0_k),
            Nbins=Nbins_histo, histo_lims=histo_lims,
            histo_title=f'{var} distribution (B0s->KK MC)', savefig=figures,
            img_name='fig/template_fit_k.png')

        res = global_fit(filepaths[2], 'tree;1', var, Nbins=Nbins_histo,
                         pars_mc1=templ_pars_k, pars_mc2=templ_pars_pi,
                         histo_lims=histo_lims, savefigs=figures)
        print(f'Frazione di K = {res.Parameters()[1]} +- {res.Errors()[1]}')
        t1 = time.time()
        print(
            f'\nTemplate fit with ROOT - ended successfully in {t1-t0} s \n\n')

    if opt in ["dnn", "all"]:
        # ~~~~~~~~ Setup of the DNN - free to edit ~~~~~~~~~~~~~~~~~~~~
        settings = dnn_settings()
        settings.layers = [75, 60, 45, 30, 20]
        settings.batch_size = 128
        settings.epochnum = 10
        settings.verbose = 2
        settings.batchnorm = False
        settings.dropout = 0.005
        settings.learning_rate = 5e-4
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("\nDeep neural network - working...\n")
        pi_eval, k_eval, data_eval = dnn(settings=settings)
        efficiency = 0.95

        y_cut, misid = find_cut(pi_eval, k_eval, efficiency)
        plt.axvline(x=y_cut, color='green', label='y cut for '
                    + str(efficiency)+' efficiency')
        plt.legend()
        plt.savefig('fig/ycut.pdf')

        rocdnnx, rocdnny, aucdnn = roc(pi_eval, k_eval, eff=efficiency,
                                       inverse_mode=False, makefig=True,
                                       name="dnn_roc")
        print("\nDeep neural network - ended successfully! \n\n")

    if opt in ["dtc", "all"]:
        print("\nTemplate fit with ROOT - working...\n")

    if opt in ["vcut", "all"]:
        print("\nTemplate fit with ROOT - working...\n")
