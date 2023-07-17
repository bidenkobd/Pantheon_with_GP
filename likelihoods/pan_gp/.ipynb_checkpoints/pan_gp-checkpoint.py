from cobaya.likelihood import Likelihood
import numpy as np
import scipy.linalg as la
from sklearn.gaussian_process.kernels import Matern
import astropy.constants
import scipy.integrate as intgr
import pandas as pn

def h_z(z,*args):
    H0,om,pwr = args
    ol  = 1 - om
    h = ( H0 * ( om * (1+z)**3 + ol ) ** 0.5 ) ** pwr
    return h 
def da(z, H0,om,pwr = -1):
    d = intgr.quad(h_z,0.,z,args=(H0,om,pwr))[0]*(astropy.constants.c.value/1000.)/(1+z)
    return d


class pan_gp(Likelihood):
    def initialize(self):

        a=np.loadtxt('/net/comas/data/users/bidenko/montepython_public/data/Pantheon/sys_full_long.dat')[1:]
        b=pn.read_csv('/net/comas/data/users/bidenko/montepython_public/data/Pantheon/lcparam_full_long_zhel.txt',delimiter = ' ')
        a= a.reshape((1048,1048))

        self.cov = a+np.diag(b['dmb'].values**2)
        
        self.mb = b['mb'].values
        self.z = b['zcmb'].values
        
        self.cccc = (self.z<0.0233)+(self.z>10.15)
        self.cond1 = np.where(self.cccc==True)
        self.cond2 = np.where(self.cccc==False)
        
        self.krn = Matern(length_scale=0.3,nu=1.5)
        self.dist = np.zeros((len(self.z),len(self.z)))
        
        for j in range(len(self.z)):
            for i in range(len(self.z)):
                self.dist[i,j] = np.abs(self.z[j]-self.z[i])
        
        self.zhel = b['zhel'].values
        self.zhel = self.zhel[self.cond2]        
        
        self.cov = np.delete(self.cov,self.cond1[0],1)
        self.cov = np.delete(self.cov,self.cond1[0],0)
        
        self.dist = np.delete(self.dist,self.cond1[0],1)
        self.dist = np.delete(self.dist,self.cond1[0],0)
        
        self.z = self.z[self.cond2]
        self.mb = self.mb[self.cond2]
        
    def get_requirements(self):
        return {}
    def logp(self, **params_values):
        
        H0gp = params_values['H0_gp']
        omegamgp = params_values['omegam_gp']
        Mgp = params_values['M_gp']
        dgp = params_values['d_gp']
        nugp = params_values['nu_gp']
        sigmagp = params_values['sigma_gp']
        
        moduli = []
        for  i in range(len(self.z)):
            moduli.append(da(self.z[i], H0gp,omegamgp,-1.) * (1. + self.z[i])*(1. + self.zhel[i]))
        moduli = np.array(moduli)
        moduli = 5 * np.log10(moduli) + 25
        
        self.krn.nu = nugp
        self.krn.length_scale = dgp
        dd= self.krn(self.dist)
        cov =self.cov.copy() + sigmagp**2 * dd

        mb = self.mb
        residuals = mb - Mgp
        residuals -= moduli

        cov = la.cholesky(cov, lower=True, overwrite_a=True)
        residuals = la.solve_triangular(cov, residuals, lower=True, check_finite=False)
     
        chi2 = (residuals**2).sum()
        return -0.5 * chi2
