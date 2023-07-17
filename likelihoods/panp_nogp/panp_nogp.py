from cobaya.likelihood import Likelihood
import numpy as np
import scipy.linalg as la
import astropy.constants
import scipy.integrate as intgr
import pandas as pn

# define functions for distance calculations
def h_z(z,*args):
    H0,om,pwr = args
    ol  = 1 - om
    h = ( H0 * ( om * (1+z)**3 + ol ) ** 0.5 ) ** pwr
    return h 
def da(z, H0,om,pwr = -1):
    d = intgr.quad(h_z,0.,z,args=(H0,om,pwr))[0]*(astropy.constants.c.value/1000.)/(1+z)
    return d

class panp_nogp(Likelihood):
    def initialize(self):
        b=pn.read_csv('./data/Pantheon+SH0ES.dat',delimiter = ' ')
        self.cov = np.reshape(pn.read_table('./data/Pantheon+SH0ES_STAT+SYS.cov',sep=' ').values,(1701,1701))
        
        self.mb = b['m_b_corr'].values
        self.z = b['zHD'].values
        self.zhel = b['zHEL'].values
        
        # define here redshift limits (e.g. to use upper redshift cut)
        self.cccc = ((b['zHD']>0.01) & (b['zHD']<10.5) | (b['IS_CALIBRATOR']==1)).values == False
        
        self.cond1 = np.where(self.cccc==True)
        self.cond2 = np.where(self.cccc==False)
        
        # applying redshift limits to datapoints, covariance and distance matrices 
        self.cov = np.delete(self.cov,self.cond1[0],1)
        self.cov = np.delete(self.cov,self.cond1[0],0)
        self.z = self.z[self.cond2]
        self.mb = self.mb[self.cond2]
        self.zhel = self.zhel[self.cond2]
         
        # defining positions of calibration SNe and distances to them
        self.sel_cal = b['IS_CALIBRATOR'].values[self.cond2]==1
        self.ceph_dist = b['CEPH_DIST'].values[self.cond2]
        self.det =  np.linalg.slogdet(self.cov.copy())[1]
        self.cov = la.cholesky(self.cov.copy(), lower=True, overwrite_a=True)


    def get_requirements(self):
        return {}
    def logp(self, **params_values):
        
        # getting current parameter from sampler
        H0gp = params_values['H0_gp']
        omegamgp = params_values['omegam_gp']
        Mgp = params_values['M_gp']

        # calculating cosmological distances
        moduli = []
        for  i in range(len(self.z)):
            moduli.append(da(self.z[i], H0gp,omegamgp) * (1. + self.z[i])*(1. + self.zhel[i]))
        moduli = np.array(moduli)
        moduli = 5 * np.log10(moduli) + 25

        # replacing distances to calibration SNe with distance measurements
        moduli[self.sel_cal] = self.ceph_dist[self.sel_cal]
        
        mb = self.mb
        residuals = mb - Mgp
        residuals -= moduli

        residuals = la.solve_triangular(self.cov.copy(), residuals, lower=True, check_finite=False)
     
        chi2 = (residuals**2).sum()
        return -0.5 * chi2 - 1./2. * self.det
