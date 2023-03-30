"""
Module that fits the MC templates with two custom functions and then fits the
data set with a weighted sum of the templates' best-fit functions obtained
"""

import ROOT
from template_fit.template_functions import DoubleGaussian, GaussJohnson
from utilities.utils import default_rootpaths


def fit_mc_template(filepath, tree, var, fitfunc, Nbins=1000,
                    histo_lims=(5.0, 5.6), histo_name='', histo_title='',
                    savefig=False, img_name='template.png'):
    """
    Performs a max-Likelihood fit on the distribution of a MC template with a
    custom function given in input

    :param filepath: Root file where the MC set (background or signal) is stored
    :type filepath: str
    :param tree: Name of the tree in Root files where the events are stored
    :type tree: str
    :param var: Variable the fit is performed on
    :type var: str
    :param fitfunc: Fit function
    :type fitfunc: ROOT::TF1
    :param Nbins: Number of bins of the histogram
    :type Nbins: int, optional
    :param histo_lims: Range of the histogram
    :type histo_lims: tuple[float]
    :param histo_name: Name of the histogram
    :type histo_name: str
    :param histo_title: Title of the histogram
    :type histo_title: str
    :param savefig: If is ``True`` saves the image of the fit
    :type savefig: bool
    :param img_name: Name of the image to be saved
    :type img_name: str
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
    Performs a max-Likelihood fit on the data set, by linearly weighting the
    templates' best-fit functions. The functional form used is fixed in the
    external library (ROOT.TemplateComposition function); the parameters
    retrieved from the fits on MC templates are used to fix this function's
    parameters, resulting in two fit parameters.

    :param filepath: Root file where the data set is stored
    :type filepath: str
    :param tree: Name of the tree in the root file where the events are stored
    :type tree: str
    :param var: Variable the fit is performed on
    :type var: str
    :param Nbins: Number of bins of the histogram
    :type Nbins: int
    :param histo_lims: Range of the histogram
    :type histo_lims: tuple[float]
    :param histo_name: Name of the histogram
    :type histo_name: str
    :param histo_title: Title of the histogram
    :type histo_name: str
    :param pars_mc1: Best-fit parameters of the first template function
    :type pars_mc1: tuple[float]
    :param pars_mc2: Best-fit parameters of the second template function
    :type pars_mc2: tuple[float]
    :param fit_range: Range where the fit is performed
    :type fit_range: tuple[float]
    :param savefig: If is ``True`` saves the image of the fit
    :type savefig: bool
    :param img_name: Name of the image to be saved
    :type img_name: str
    :return: Root object containing the fit details
    :rtype: ROOT::TFitResult
    """
    if histo_name == '':
        histo_name = var
    if histo_title == '':
        histo_title = var + ' distribution (data)'

    ROOT.gStyle.SetOptFit(11111)
    histo_title = var + ' distribution (data)'

    df = ROOT.RDataFrame(tree, filepath)

    h = df.Histo1D(
        (var, histo_title, Nbins, histo_lims[0], histo_lims[1]), var)

    npars_1 = len(pars_mc1)
    npars_2 = len(pars_mc2)
    npars_tot = 2 + npars_1 + npars_2

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
    print('Running this module as main module is not supported. Feel free to \
add a custom main or run the package as a whole (see README.md)')
