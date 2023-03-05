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
from utilities.gen_from_toy import gen_from_toy
from utilities.utils import default_rootpaths, default_vars
from template_fit.template_fit import fit_templates
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

parser.add_argument('-rp', '--rootpaths', nargs=2, default=default_toyMC_path,
                    help='Path of the toyMC root files taken as input')

parser.add_argument('-tr', '--tree', default='t_M0pipi;1',
                    help='Name of the tree of the toyMC root files where the variables are stored')

parser.add_argument('-g', '--generate', action='store_true',
                    help='Generates MC and mixed datasets from the toyMC')

parser.add_argument('-rpo', '--rootpaths_out', nargs=3, default=default_rootpaths(),
                    help='Path of the datasets generated from the toyMC')

parser.add_argument('-fp', '--figpath',
                    help='Path where figures are going to be saved')

parser.add_argument('-v', '--variables', nargs='+', default=default_vars(),
                    help='Variables you want to treat')

parser.add_argument('-vfit', '--var_fit', default='M0_Mpipi',
                    help='Variable on which template fit is performed')

parser.add_argument('-f', '--fraction', type=float, default=0.42,
                    help='Fraction of Kaons in the mixed dataset you want to generate')

parser.add_argument('-t', '--type', nargs='+', default='all',
                    choices=['tfit', 'dnn', 'dtc', 'vcut', 'all'],
                    help='Type of the analysis to be performed')

parser.add_argument('-cr', '--cornerplot', action='store_true',
                    help='Choice to generate and save the cornerplot')

args = parser.parse_args()


filepaths = args.rootpaths
figpath = args.figpath
tree = args.tree

var_fit = args.var_fit

if args.generate:
    gen_from_toy(filepaths_in=tuple(args.rootpaths), tree=tree, f=args.fraction,
                 num_mc=100000, num_data=30000, vars=tuple(args.variables))


if 'all' in args.type:
    analysis = ['all']
else:
    analysis = []
    # Remove duplicates from analysis methods list
    [analysis.append(item) for item in args.type if item not in analysis]

for opt in analysis:

    if opt == "tfit" or opt == "all":
        print("\nTemplate fit with ROOT - working...\n")
        res = fit_templates(args.rootpaths_out, 'tree;1', var_fit)
        print(f'Frazione di K = {res.Parameters()[1]} +- {res.Errors()[1]}')
        print("\nTemplate fit with ROOT - ended successfully! \n\n")

    if opt == "dnn" or opt == "all":
        print("\nTemplate fit with ROOT - working...\n")

    if opt == "dtc" or opt == "all":
        print("\nTemplate fit with ROOT - working...\n")

    if opt == "vcut" or opt == "all":
        print("\nTemplate fit with ROOT - working...\n")


print(type(args.type))
print(args.fraction)
