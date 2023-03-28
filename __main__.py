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
import os
import argparse
import time
import numpy as np
from random import randint
# import matplotlib.pyplot as plt
from machine_learning.deepnn import dnn
from machine_learning.dtc import dt_classifier
from template_fit.template_fit import fit_mc_template, global_fit
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from utilities.cornerplot import overlaid_cornerplot
from utilities.gen_from_toy import gen_from_toy
from utilities.dnn_settings import DnnSettings
from utilities.utils import default_rootpaths, default_resultsdir, \
                            default_vars, find_cut, roc, plot_rocs
from var_cut.var_cut import var_cut
import warnings

warnings.formatwarning = lambda msg, *args, **kwargs: f'\n{msg}\n'

# print(" ----------------------------------------------- ")
# print("|  Welcome to the PiK Classifier package!       |")
# print("|                                               |")
# print("|  Authors: Lorenzo Punzi, Ruben Forti          |")
# print("|  Release: 1.0  -  march 2023                  |")
# print(" ----------------------------------------------- ")


default_toyMC_path = ('cmepda-PiKclassifier/data/root_files/toyMC_B0PiPi.root',
                      'cmepda-PiKclassifier/data/root_files/toyMC_B0sKK.root')

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

parser.add_argument('-fig', '--figures', action='store_true',
                    help='Saves the generated figures')

parser.add_argument('-rd', '--resdir', default=default_resultsdir(dir='cmepda-PiKclassifier/outputs_main'),
                    help='Directory where to save the results')

parser.add_argument('-cr', '--cornerplot', action='store_true',
                    help='Generates and saves the cornerplot of the variables')


# ~~~~~~~~ Subparser for datasets generation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser_gen = subparsers.add_parser(
    'gen', help='Generates MC and mixed datasets from the toyMC')

parser_gen.add_argument('-rpt', '--rootpaths_toy', nargs=2, default=default_toyMC_path,
                        help='Path of the toyMC root files taken as input')

parser_gen.add_argument('-ndat', '--num_events', nargs=2, default=[0,0], type=int,
                        help='Number of events in each MC dataset and in the mixed one (in this order)')

parser_gen.add_argument('-f', '--fraction', type=float, default=0.42,
                        help='Fraction of Kaons in the mixed dataset you want to generate')


# ~~~~~~~~ Subparser for analysis options ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

parser_an = subparsers.add_parser('analysis', help='Performs the analyses')

parser_an.add_argument('-m', '--methods', nargs='+', default='all',
                       choices=['tfit', 'dnn', 'dtc', 'vcut', 'all'],
                       help='Type of the analysis to be performed. \
                             If \'all\' is called, the default variable for the ROOT template fit and the var_cut are selected')

parser_an.add_argument('-vfit', '--var_fit', default='M0_Mpipi',
                       help='Variable on which template fit is performed')

parser_an.add_argument('-ld', '--load_dnn', action='store_true',
                       help='Loads DNN model and weights saved as .json and .h5 files')

parser_an.add_argument('-vcut', '--var_cut', nargs='+', default='M0_Mpipi',
                       help='Variable(s) on which cut evaluation is performed')
'''
parser_an.add_argument('-inv', -'vcut_inverse', nargs='+', default=True,
                       help='Flag(s) that select the inverse mode to perform the variable(s) cut analysis')
'''
parser_an.add_argument('-e', '--efficiency', type=float, default=0.90,
                       help='Probability of correct K identification requested (applies only to dnn and var_cut analyses)')

parser_an.add_argument('-su', '--stat_uncertainties', action='store_true',
                       help='Performs the statistical analysis for the methods selected')


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


args = parser.parse_args()

filepaths = args.rootpaths_gen
tree = args.tree

respath = os.path.join(os.getcwd(), args.resdir)

results_file = os.path.join(respath, 'results.txt')
with open(results_file, encoding='utf-8', mode='w') as f:
    f.write('\n Results of the analysis performed with PiK classifier package \n'
            ' ------------------------------------------------------------- \n\n')


def add_result(name, value, note=''):
    """
    """
    with open(results_file, encoding='utf-8', mode='a') as file:
        if note == '':
            file.write(f'    {name} = {value}\n')
        else:
            file.write(f'    {name} = {value}  |  {note} \n')


if hasattr(args, 'rootpaths_toy'):
    # Generates the datasets with the requested fraction of Kaons in
    # the mixed sample. If the following two quantities are BOTH set to
    # zero, the function generates the datasets with the maximum
    # possible number of events
    NUM_MC, NUM_DATA = args.num_events
    print(NUM_MC, NUM_DATA)
    gen_from_toy(filepaths_in=tuple(args.rootpaths_toy), tree=args.tree,
                 fraction=args.fraction, vars=tuple(args.variables),
                 num_mc=NUM_MC, num_data=NUM_DATA)


if args.cornerplot is True:
    if args.variables == default_vars():
        overlaid_cornerplot(vars=args.variables[:5], figpath=respath)
        overlaid_cornerplot(vars=args.variables[7:], figpath=respath)
        overlaid_cornerplot(
            vars=('M0_p',)+args.variables[7:9], figpath=respath)
    elif len(args.variables) > 5:
        msg = '***WARNING*** \nNumber of variables to print in the corner plot\
        exceeds the maximum suggested (5). Running the \'cornerplot\' function\
        on groups of five contiguous variables in the list.'
        warnings.warn(msg, stacklevel=2)
        ind = 0
        while ind+5 <= len(args.variables):
            overlaid_cornerplot(
                vars=args.variables[ind:ind+4], figpath=respath)
            ind += 5
        overlaid_cornerplot(vars=args.variables[ind:], figpath=respath)


if hasattr(args, "methods"):
    # Initialize a list with the requesteds method of analysis, also removing
    # duplicates
    SINGULAR_ROCS = True
    if 'all' in args.methods:
        analysis = ['all']
        SINGULAR_ROCS = False
    else:
        analysis = []
        [analysis.append(item) for item in args.methods
            if item not in analysis]
        roc_analysis = ["dnn", "dtc", "vcut"]
        flag = len([a for a in analysis if a in roc_analysis])
        if flag >= 2 or len([args.var_cut]) > 1:
            SINGULAR_ROCS = False

    if SINGULAR_ROCS is not True:
        rocx_array, rocy_array = [], []
        roc_labels, roc_linestyles, roc_colors = [], [], []
        x_pnts, y_pnts, point_labels = [], [], []

    for opt in analysis:

        if opt in ["tfit", "all"]:
            # ~~~~~~~~ Setup of the template fit - free to edit ~~~~~~~~~~~~~~
            NBINS_HISTO = 1000
            histo_lims = (5.0, 5.6)  # Limits of the histograms
            fit_range = (5.02, 5.42)  # Range where the templates are fitted
            p0_pi = (1e5, 0.16, 5.28, 0.08, 5.29, 0.04)
            p0_k = (1e5, 0.97, 1.6, 0.046, 5.30, 1.1, 5.27, 0.00045)
            figures = args.figures
            FIGNAME_TEMPL_PI = 'Template_fit_Pi.pdf'
            FIGNAME_TEMPL_K = 'Template_fit_K.pdf'
            FIGNAME_GLOBAL = 'Template_fit_Data.pdf'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            t0 = time.time()
            method_title = 'Template fit with ROOT'
            with open(results_file, encoding='utf-8', mode='a') as file_tfit:
                file_tfit.write(f'\n\n  {method_title}: \n')
            print(f'\n {method_title} - working...\n')
            var = args.var_fit

            templ_pars_pi = fit_mc_template(
                filepaths[0], args.tree, var,
                DoubleGaussian(fit_range, pars=p0_pi),
                Nbins=NBINS_HISTO, histo_lims=histo_lims,
                histo_title=f'{var} distribution (B0->PiPi MC)',
                savefig=figures, img_name=f'{respath}/{FIGNAME_TEMPL_PI}')
            templ_pars_k = fit_mc_template(
                filepaths[1], args.tree, var,
                GaussJohnson(fit_range, pars=p0_k),
                Nbins=NBINS_HISTO, histo_lims=histo_lims,
                histo_title=f'{var} distribution (B0s->KK MC)',
                savefig=figures, img_name=f'{respath}/{FIGNAME_TEMPL_K}')

            res = global_fit(filepaths[2], args.tree, var, Nbins=NBINS_HISTO,
                             pars_mc1=templ_pars_k, pars_mc2=templ_pars_pi,
                             histo_lims=histo_lims, savefig=figures,
                             img_name=f'{respath}/{FIGNAME_GLOBAL}')

            add_result(
                "K fraction", f'{res.Parameters()[1]} +- {res.Errors()[1]}')
            add_result("Chi2", res.Chi2())
            add_result("Probability", res.Prob())
            t1 = time.time()
            print(
                f'  {method_title} - ended successfully in {t1-t0} s \n\n')

        if opt in ["dnn", "all"]:
            # ~~~~~~~~ Setup of the DNN - free to edit ~~~~~~~~~~~~~~~~~~~~~~~
            settings = DnnSettings()
            settings.layers = (75, 60, 45, 30, 20)
            # settings.val_fraction = 0.5
            settings.batch_size = 128
            settings.epochnum = 400
            settings.verbose = 2
            settings.dropout = 0
            settings.learning_rate = 5e-4
            MODEL_FILE = 'cmepda-PiKclassifier/machine_learning/deepnn.json'
            WEIGHTS_FILE = 'cmepda-PiKclassifier/machine_learning/deepnn.h5'
            INVERSE = False
            figs = args.figures
            PLOT_ROC = bool(args.figures*SINGULAR_ROCS)
            fignames = ("DNN_history.png", "eval_Pions.png", "eval_Kaons.png",
                        "eval_Data.png")
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            method_title = 'Deep Neural Network'
            with open(results_file, encoding='utf-8', mode='a') as file_dnn:
                file_dnn.write(f'\n\n  {method_title}: \n')
            print(f'\n  {method_title} - working...\n')

            fr, stats, eval_test = dnn(
                source=('root', filepaths), root_tree=tree,
                vars=args.variables, settings=settings, load=args.load_dnn,
                trained_filenames=(MODEL_FILE, WEIGHTS_FILE),
                efficiency=args.efficiency,
                savefigs=figs, figpath=respath, fignames=fignames)

            rocx_dnn, rocy_dnn, auc_dnn = roc(
                eval_test[0], eval_test[1], eff=stats[2], inverse_mode=INVERSE,
                makefig=PLOT_ROC, name=f'{respath}/ROC_dnn')

            if SINGULAR_ROCS is not True:
                rocx_array.append(rocx_dnn)
                rocy_array.append(rocy_dnn)
                roc_labels.append("DNN")

            add_result(
                "K fraction", f'{round(fr[0],4)} +- {round(fr[1],4)} (stat) +- {round(fr[2], 4)} (syst)')
            add_result("Efficiency", stats[0])
            add_result("Misid", stats[1])
            add_result("Cut", stats[2])
            add_result("AUC", auc_dnn)
            print(f"\n  {method_title} - ended successfully! \n\n")

        if opt in ["dtc", "all"]:
            # ~~~~~~~~ Setup of the DTC - free to edit ~~~~~~~~~~~~~~~~~~~~~~~
            TEST_SIZE = 0.3
            ML_SAMP = 1
            CRIT = 'gini'
            PRINTED_TREE_FILE = 'DTC_printed.txt'
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            method_title = 'Decision Tree Classifier'
            with open(results_file, encoding='utf-8', mode='a') as file_dtc:
                file_dtc.write(f'\n\n  {method_title}: \n')
            print(f'\n  {method_title} - working...\n')

            fr, stats = dt_classifier(
                root_tree=args.tree, vars=args.variables, test_size=TEST_SIZE,
                min_leaf_samp=ML_SAMP, crit=CRIT,
                print_tree=f'{respath}/{PRINTED_TREE_FILE}', figpath=respath)

            if SINGULAR_ROCS is not True:
                x_pnts.append(stats[1])
                y_pnts.append(stats[0])
                point_labels.append("DTC")

            add_result(
                "K fraction", f'{round(fr[0],4)} +- {round(fr[1],4)} (stat) +- {round(fr[2], 4)} (syst)')
            add_result("Efficiency", stats[0])
            add_result("Misid", stats[1])
            print(f'\n  {method_title} - ended successfully! \n\n')

        if opt in ["vcut", "all"]:
            # ~~~~~~~~ Setup of the var_cut - free to edit ~~~~~~~~~~~~~~~~~~~
            INVERSE = True
            SPECIFICITY = False
            figure_cut = args.figures
            figure_roc = bool(args.figures*SINGULAR_ROCS)
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            type_title = 'Cut on Variables Distribution'
            with open(results_file, encoding='utf-8', mode='a') as file_vcut:
                file_vcut.write(f'\n\n  {type_title}: \n')
            print(f'\n  {type_title} - working...\n')

            rocx_vcut, rocy_vcut, labels_vcut = [], [], []

            print(args.var_cut)
            for vc in [args.var_cut]:
                fr, stats, eval_arr = var_cut(
                    rootpaths=filepaths, tree=tree, cut_var=vc, eff=args.efficiency,
                    inverse_mode=INVERSE, specificity_mode=SPECIFICITY,
                    draw_fig=figure_cut, figpath=respath)

                rocx, rocy, auc = roc(
                    eval_arr[0], eval_arr[1], eff=round(stats[0], 4), inverse_mode=INVERSE,
                    makefig=figure_roc, name=f'{respath}/ROC_{vc}_cut')

                add_result(
                    "K fraction", f'{round(fr[0],4)} +- {round(fr[1],4)} (stat) +- {round(fr[2],4)} (syst)', vc)
                add_result("Efficiency", stats[0], vc)
                add_result("Misid", stats[1], vc)
                add_result("Cut", stats[2], vc)
                add_result("AUC", auc, vc)
                if SINGULAR_ROCS is not True:
                    rocx_array.append(rocx)
                    rocy_array.append(rocy)
                    roc_labels.append(f'{vc}')
            print(f"\n  {type_title} - ended successfully! \n\n")

    if SINGULAR_ROCS is not True:
        for i in range(len(roc_labels)):
            roc_colors.append('#%06X' % randint(0, 0xFFFFFF))
            roc_linestyles.append('-')
        plot_rocs(tuple(rocx_array), tuple(rocy_array), tuple(roc_labels),
                  tuple(roc_linestyles), tuple(roc_colors),
                  x_pnts=x_pnts, y_pnts=y_pnts, point_labels=point_labels,
                  eff=args.efficiency, figtitle='ROCs', figname=f'{respath}/ROCs.png')


print("END OF FILE")
