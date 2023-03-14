"""
Module stores the functions able to perform fit on MC templates and to weight
them in a template fit to retrieve an estimate of the fraction f of kaons in
the mixed Pi-K population present in the dataset Bhh_data.root
"""

import ROOT
import time
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from utilities.utils import default_rootpaths


def fit_mc_template(filepath, tree, var, fitfunc, Nbins=1000,
                    histo_lims=(5.0, 5.6), histo_name='', histo_title='',
                    savefig=False, img_name='template.png'):
    """
    Performs a max-Likelihood fit on the distribution of a MC sample

    :param filepath: Root file where the MC dataset is stored
    :type filepath: tuple[str]
    :param tree: Name of the tree in root file where the events are stored
    :type tree: str
    :param var: Variable the fit is performed on
    :type var: str
    :param fitfunc: Fit function
    :type fitfunc: ROOT::TF1
    :param Nbins: Number of bins of the histogram, defaults to 1000
    :type Nbins: int, optional
    :param histo_lims: Range of the histogram, defaults to (5.0, 5.6)
    :type histo_lims: tuple[float], optional
    :param histo_name: Name of the histogram
    :type histo_name: str, optional
    :param histo_title: Title of the histogram
    :type histo_name: str, optional
    :param savefig: If is ``True`` saves the image of the fit, defaults to ``False``
    :type savefig: bool, optional
    :param img_name: Name of the image to be saved
    :type img_name: str, optional
    :return: Best-fit farameters of the fit function
    :rtype: tuple[float]
    """
    if histo_name == '':
        histo_name = var
    if histo_title == '':
        histo_title = var + ' distribution (MC)'

    df = ROOT.RDataFrame(tree, filepath)

    h = df.Histo1D((histo_name, histo_title, Nbins,
                   histo_lims[0], histo_lims[1]), var)

    ROOT.gStyle.SetOptFit(11111)

    if savefig is True:
        fitvar = h.Fit(fitfunc, "SQLR")
        c = ROOT.TCanvas()
        c.cd()
        h.Draw()
        fitfunc.Draw("same")
        c.SaveAs(img_name)
    else:
        fitvar = h.Fit(fitfunc, "SQLRN")

    print('Fit on ' + histo_title + ' - stats:')
    print(f'  Chi2 = {fitvar.Chi2()}')
    print(f'  Ndof = {fitvar.Ndf()}')
    print(f'  Prob = {fitvar.Prob()}\n')

    exit_pars = []
    for k in range(fitfunc.GetNpar()):
        exit_pars.append(fitfunc.GetParameter(k))

    return tuple(exit_pars)


def global_fit(filepath, tree, var, Nbins=1000, histo_lims=(5.0, 5.6),
               histo_name='', histo_title='', pars_mc1=(), pars_mc2=(),
               fit_range=(5.02, 5.42), savefig=False, img_name="fig/Fit_data.png"):
    """
    Performs a max-Likelihood fit on the mixed dataset, by weighting (linearly)
    the template functions found for the two MC import_datasets. The template
    functions used in this case are already mixed in the function
    ROOT.TemplateComposition (external library) and the parameters given by the
    fit on MC datasets are passed into this function.

    :param filepath: Root file where the MC dataset is stored
    :type filepath: tuple[str]
    :param tree: Name of the tree in root file where the events are stored
    :type tree: str
    :param var: Variable the fit is performed on
    :type var: str
    :param Nbins: Number of bins of the histogram, defaults to 1000
    :type Nbins: int, optional
    :param histo_lims: Range of the histogram, defaults to (5.0, 5.6)
    :type histo_lims: tuple[float], optional
    :param histo_name: Name of the histogram
    :type histo_name: str, optional
    :param histo_title: Title of the histogram
    :type histo_name: str, optional
    :param pars_mc1: Best-fit parameters for the first template function
    :type pars_mc1: tuple[float]
    :param pars_mc2: Best-fit parameters for the second template function
    :type pars_mc2: tuple[float]
    :param fit_range: Range where the fit is performed
    :type fit_range: tuple[float]
    :param savefig: If is ``True`` saves the image of the fit, defaults to ``False``
    :type savefig: bool, optional
    :param img_name: Name of the image to be saved
    :type img_name: str, optional
    :return: Best-fit farameters of the fit function
    :rtype: tuple[float]
    """
    if histo_name == '':
        histo_name = var
    if histo_title == '':
        histo_title = var + ' distribution (data)'

    npars_1 = len(pars_mc1)
    npars_2 = len(pars_mc2)
    npars_tot = 2 + npars_1 + npars_2

    df = ROOT.RDataFrame(tree, filepath)

    # ROO T.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptFit(11111)
    histo_title = var + ' distribution (data)'
    h = df.Histo1D(
        (var, histo_title, Nbins, histo_lims[0], histo_lims[1]), var)

    f_data = ROOT.TF1("f_data", ROOT.TemplateComposition,
                      fit_range[0], fit_range[1], npars_tot)

    [f_data.FixParameter(2+k, pars_mc1[k]) for k in range(npars_1)]
    [f_data.FixParameter(2+npars_1+k, pars_mc2[k]) for k in range(npars_2)]

    if savefig is True:
        fitvar = h.Fit(f_data, "SQLR")
        c = ROOT.TCanvas()
        c.cd()
        h.Draw()
        f_data.Draw("same")
        c.SaveAs(img_name)
    else:
        fitvar = h.Fit(f_data, "SQLRN")

    print('Fit on ' + histo_title + ' - stats:')
    print(f'  Chi2 = {fitvar.Chi2()}')
    print(f'  Ndof = {fitvar.Ndf()}')
    print(f'  Prob = {fitvar.Prob()}\n')

    return fitvar


if __name__ == "__main__":
    t0 = time.time()
    filepaths = default_rootpaths()
    tree = 'tree;1'
    var = 'M0_Mpipi'
    histo_lims = (5.0, 5.6)
    fit_range = (5.02, 5.42)

    templ_pars_pi = fit_mc_template(filepaths[0], tree, var,
                                    DoubleGaussian(fit_range),
                                    img_name='fig/template_fit_pi.png',
                                    histo_title=f'{var} distribution (B0->PiPi MC)',
                                    savefig=True)

    templ_pars_k = fit_mc_template(filepaths[1], tree, var,
                                   GaussJohnson(fit_range),
                                   img_name='fig/template_fit_k.png',
                                   histo_title=f'{var} distribution (B0s->KK MC)',
                                   savefig=False)

    results = global_fit(filepaths[2], tree, var, pars_mc1=templ_pars_k,
                         pars_mc2=templ_pars_pi, savefigs=False)

    print(
        f'Frazione di K = {results.Parameters()[1]} +- {results.Errors()[1]}')

    t1 = time.time()
    print(f'Elapsed time = {t1-t0} s')
