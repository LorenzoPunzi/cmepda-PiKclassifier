
import ROOT
from ROOT import TCanvas, TColor
from ROOT import kGreen, kBlue


def SetCanva(Name, Title, width=900, heigth=600, residuals=True):
    """
    Function that creates a new Canva (with residual subplot option)
    """
    c = ROOT.TCanvas(Name, Title, width, heigth)

    if residuals:
        pad1 = ROOT.TPad("", "", 0, 0.33, 1, 1)
        pad2 = ROOT.TPad("", "", 0, 0, 1, 0.33)

        pad1.SetBottomMargin(0.001)
        pad1.SetBorderMode(0)
        pad2.SetBottomMargin(0.25)
        pad2.SetBorderMode(0)

        pad1.Draw()
        pad2.Draw()

    return c


def DrawLegend(*elements, legTitle="Legend Title"):
    """
    """
    if len(elements) == 0:
        pass

    legend = ROOT.TLegend(0.2, 0.2)
    legend.SetHeader(legTitle, "C")
    [legend.AddEntry(el) for el in elements]
    return legend


def DrawPlot():
    """
    """


if __name__ == '__main__':
    c1 = SetCanva("", "")
    h = ROOT.TH1F("h", "h", 10, -1, 1)
    h.FillRandom("gaus", 1000)
    pad1.cd()
    h.Draw()
    leg = DrawLegend(h, legTitle="Title")
    leg.Draw("same")
    c1.SaveAs("provaplot.png")
