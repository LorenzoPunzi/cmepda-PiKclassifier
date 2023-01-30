#include "header.h"


using namespace ROOT;

double Gaus(double* x, double* par) {
    return TMath::Exp(-(x[0]-par[0])*(x[0]-par[0])/(2*par[1]))/TMath::Sqrt(2*TMath::Pi()*par[1]*par[1]);
}


double Johnson(double* x, double* par) {
    double c = 1/TMath::Sqrt(2*TMath::Pi());
    double arg = (x[0]-par[2])/par[1];
    double expo_arg = (par[3]+par[0]*TMath::ASinH(arg));
    double expo = TMath::Exp(-0.5*expo_arg*expo_arg);
    double denom = TMath::Sqrt(1+(arg*arg));
    return c*(par[0]/par[1])*expo/denom;
}


double DoubleGaussian(double* x, double* par) {
    // In total has 6 parameters
    return (6e-4)*par[0]*(par[1]*Gaus(x, &par[2]) + (1-par[1])*Gaus(x, &par[4]));
}


double GaussJohnson(double* x, double* par) {
    // In total has 8 parameters
    return (6e-4)*par[0]*(par[1]*Johnson(x, &par[2]) + (1-par[1])*Gaus(x, &par[6]));
}


double TemplateComposition(double* x, double* par) {
    // In total has 16 parameters
    return par[0]*(par[1]*GaussJohnson(x, &par[2]) + (1-par[1])*DoubleGaussian(x, &par[6]));
}


int main() {
    std::cout << "Ciao" << std::endl;
    return 0;
}
