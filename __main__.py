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
from machine_learning.dtc import dt_classifier
from template_fit.template_fit_var import fit_mc_template, global_fit
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from utilities.gen_from_toy import gen_from_toy
from utilities.dnn_settings import dnn_settings
from utilities.utils import default_rootpaths, default_txtpaths, default_vars,\
                            find_cut, roc, plot_rocs
from var_cut.var_cut import var_cut

# print(" ----------------------------------------------- ")
# print("|  Welcome to the PiK Classifier package!       |")
# print("|                                               |")
# print("|  Authors: Lorenzo Punzi, Ruben Forti          |")
# print("|  Release: 1.0  -  march 2023                  |")
# print(" ----------------------------------------------- ")


def add_result(name, value, err=0, *note):
    with open('results.txt', encoding='utf-8', mode='a') as file:
        if err == 0:
            file.write(f'    {name} = {value}  |  {note}\n')
        else:
            file.write(f'    {name} = {value} +- {err}  |  {note}\n')


default_toyMC_path = ('data/root_files/toyMC_B0PiPi.root',
                      'data/root_files/toyMC_B0sKK.root')

parser = argparse.ArgumentParser(prog='PiK classifier',
                                 description='What the program does',
                                 epilog='Text at the bottom of help')

subparsers = parser.add_subparsers(help='Sub-command help')


# ~~~~~~~~ Generic arguments for the main parser ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser.add_argument('-tr', '--tree', default='t_M0pipi;1',
                    help='Name of the tree where the variables are stored (valid both for the toyMCs and the generated_datasets)')

parser.add_argument('-rpg', '--rootpaths_gen', nargs=3, default=default_rootpaths(),
                    help='Path of the datasets on which to perform the analysis')

parser.add_argument('-v', '--variables', nargs='+', default=default_vars(),
                    help='Variables you want to treat')


# ~~~~~~~~ Subparser for datasets generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser_gen = subparsers.add_parser(
    'gen', help='Generates MC and mixed datasets from the toyMC')

parser_gen.add_argument('-rpt', '--rootpaths_toy', nargs=2, default=default_toyMC_path,
                        help='Path of the toyMC root files taken as input')

parser_gen.add_argument('-ndat', '--num_events', nargs=2, default=0, type=int,
                        help='Number of events in each MC dataset and in the mixed one (in this order)')

parser_gen.add_argument('-f', '--fraction', type=float, default=0.42,
                        help='Fraction of Kaons in the mixed dataset you want to generate')


# ~~~~~~~~ Subparser for analysis options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser_an = subparsers.add_parser('analysis', help='Performs the analysis')

parser_an.add_argument('-m', '--methods', nargs='+', default='all',
                       choices=['tfit', 'dnn', 'dtc', 'vcut', 'all'],
                       help='Type of the analysis to be performed. \n'
                            'If \'all\' is called, the default variable for the ROOT template fit and the var_cut are selected')

parser_an.add_argument('-vfit', '--var_fit', default='M0_Mpipi',
                       help='Variable on which template fit is performed')

parser_an.add_argument('-vcut', '--var_cut', nargs='+', default='M0_Mpipi',
                       help='Variable(s) on which cut evaluation is performed')

parser_an.add_argument('-e', '--efficiency', default=0.90,
                       help='Probability of correct Pi identification requested (applies only to dnn and var_cut analyses)')


# ~~~~~~~~ Subparser for plots options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

'''
parser_fig = subparsers.add_parser('figures', help='Saves the figures')

parser_fig.add_argument('-fp', '--figpath',
                        help='Path where figures are going to be saved')

parser_fig.add_argument('-cr', '--cornerplot', action='store_true',
                        help='Choice to generate and save the cornerplot of the variables given in input')
'''


args = parser.parse_args()

filepaths = args.rootpaths_gen
# figpath = args.figpath
tree = args.tree

with open('results.txt', encoding='utf-8', mode='w') as f:
    f.write('\n Results of the analysis performed with PiK classifier package \n'
            ' ------------------------------------------------------------- \n\n')


if hasattr(args, 'rootpaths_toy'):
    # Generates the datasets with the requested fraction of Kaons in
    # the mixed sample. If the following two quantities are BOTH set to
    # zero, the function generates the datasets with the maximum
    # possible number of events
    NUM_MC, NUM_DATA = args.num_events
    print(NUM_MC, NUM_DATA)
    gen_from_toy(filepaths_in=tuple(args.rootpaths_toy), tree=args.tree,
                 f=args.fraction, vars=tuple(args.variables),
                 num_mc=NUM_MC, num_data=NUM_DATA)


if hasattr(args, "methods"):
    # Initialize a list with the requesteds method of analysis, also removing duplicates
    if 'all' in args.methods:
        analysis = ['all']
    else:
        analysis = []
        a = [analysis.append(item)
             for item in args.methods if item not in analysis]

    for opt in analysis:

        if opt in ["tfit", "all"]:
            # ~~~~~~~~ Setup of the template fit - free to edit ~~~~~~~~~~~~~~
            Nbins_histo = 1000
            histo_lims = (5.0, 5.6)  # Limits of the histograms
            fit_range = (5.02, 5.42)  # Range where the functions are fitted
            p0_pi = (1e5, 0.14, 5.28, 0.08, 5.29, 0.02)
            p0_k = (1e05, 0.991, 1.57, 0.045, 5.29, 1.02, 5.28, 0.00043)
            figures = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            t0 = time.time()
            type_title = 'Template fit with ROOT'
            with open('results.txt', encoding='utf-8', mode='a') as f:
                f.write(f'\n\n  {type_title}: \n')
            print(f'\n {type_title} - working...\n')
            var = args.var_fit

            templ_pars_pi = fit_mc_template(
                filepaths[0], args.tree, var,
                DoubleGaussian(fit_range, pars=p0_pi),
                Nbins=Nbins_histo, histo_lims=histo_lims,
                histo_title=f'{var} distribution (B0->PiPi MC)',
                savefig=figures, img_name='fig/template_fit_pi.png')
            templ_pars_k = fit_mc_template(
                filepaths[1], args.tree, var,
                GaussJohnson(fit_range, pars=p0_k),
                Nbins=Nbins_histo, histo_lims=histo_lims,
                histo_title=f'{var} distribution (B0s->KK MC)',
                savefig=figures, img_name='fig/template_fit_k.png')

            res = global_fit(filepaths[2], args.tree, var, Nbins=Nbins_histo,
                             pars_mc1=templ_pars_k, pars_mc2=templ_pars_pi,
                             histo_lims=histo_lims, savefigs=figures)

            add_result("K fraction", res.Parameters()[1], err=res.Errors()[1])
            add_result("Chi2", res.Chi2())
            add_result("Probability", res.Prob())
            t1 = time.time()
            print(
                f'  {type_title} - ended successfully in {t1-t0} s \n\n')

        if opt in ["dnn", "all"]:
            # ~~~~~~~~ Setup of the DNN - free to edit ~~~~~~~~~~~~~~~~~~~~~~~
            settings = dnn_settings()
            settings.layers = [75, 60, 45, 30, 20]
            settings.batch_size = 128
            settings.epochnum = 100
            settings.verbose = 2
            settings.batchnorm = False
            settings.dropout = 0.005
            settings.learning_rate = 5e-4
            inverse = False
            figures = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            type_title = 'Deep Neural Network'
            with open('results.txt', encoding='utf-8', mode='a') as file:
                file.write(f'\n\n  {type_title}: \n')
            print(f'\n  {type_title} - working...\n')
            pi_eval, k_eval, data_eval = dnn(settings=settings)
            y_cut, misid = find_cut(pi_eval, k_eval, args.efficiency)
            # plt.axvline(x=y_cut, color='green', label='y cut for '
            #             + str(efficiency)+' efficiency')
            # plt.legend()
            #   plt.savefig('fig/ycut.pdf')
            rocdnnx, rocdnny, aucdnn = roc(pi_eval, k_eval, eff=args.efficiency,
                                           inverse_mode=inverse, makefig=figures,
                                           name="dnn_roc")
            fraction = ((data_eval > y_cut).sum()
                        / data_eval.size-misid)/(args.efficiency-misid)
            add_result("K fraction", fraction)
            add_result("Output cut", y_cut, f'Efficiency = {args.efficiency}')
            add_result("Misid", misid, f'Efficiency = {args.efficiency}')
            add_result("AUC", aucdnn, f'Efficiency = {args.efficiency}')
            print(f"\n  {type_title} - ended successfully! \n\n")

        if opt in ["dtc", "all"]:
            # ~~~~~~~~ Setup of the DTC - free to edit ~~~~~~~~~~~~~~~~~~~~~~~
            test_size = 0.3,
            ml_samp = 1,
            crit = 'gini'
            print = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            dtc_title = 'Decision Tree Classifier'
            with open('results.txt', encoding='utf-8', mode='a') as file:
                file.write(f'\n\n  {dtc_title}: \n')
            # print(f'\n  {dtc_title} - working...\n')
            pred_array, eff, misid = dt_classifier(
                root_tree=args.tree, vars=args.variables, test_size=test_size,
                min_leaf_samp=ml_samp, crit=crit, print_tree=print)
            fraction = pred_array.sum()/len(pred_array)
            add_result("K fraction", fraction)
            add_result("Efficiency", eff)
            add_result("Misid", misid)
            print(f"\n  {dtc_title} - ended successfully! \n\n")

        if opt in ["vcut", "all"]:
            # ~~~~~~~~ Setup of the var_cut - free to edit ~~~~~~~~~~~~~~~~~~~
            inverse = False
            specificity = False
            figure = False
            roc_figure = False
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            type_title = 'Cut on Variables Distribution'
            with open('results.txt', encoding='utf-8', mode='a') as file:
                file.write(f'\n\n  {type_title}: \n')
            print(f'\n  {type_title} - working...\n')
            fraction, misid, rocx, rocy, auc = var_cut(
                rootpaths=filepaths, tree=tree, cut_var=args.var_cut,
                eff=args.efficiency, inverse_mode=inverse, specificity_mode=specificity,
                draw_roc=roc_figure, draw_fig=figure)
            add_result("K fraction", fraction, f'{(args.var_cut)}')
            add_result("Misid", misid, f'{(args.var_cut)}')
            add_result("AUC", auc, f'{(args.var_cut)}')
            print(f"\n  {type_title} - ended successfully! \n\n")


print("END OF FILE")
