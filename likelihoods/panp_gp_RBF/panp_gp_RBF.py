from cobaya.likelihood import Likelihood
import numpy as np
import scipy.linalg as la
from sklearn.gaussian_process.kernels import RBF
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

class panp_gp_RBF(Likelihood):
    def initialize(self):
        
        b=pn.read_csv('/net/comas/data/users/bidenko/cobaya/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',delimiter = ' ')
        self.cov = np.reshape(pn.read_table(r'/net/comas/data/users/bidenko/cobaya/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov',sep=' ').values,(1701,1701))
        
        
        self.mb = b['m_b_corr'].values
        self.z = b['zHD'].values
        self.zhel = b['zHEL'].values
        
        # define here redshift limits (e.g. to use upper redshift cut)
        self.cccc = ((b['zHD']>0.01) & (b['zHD']<10.5) | (b['IS_CALIBRATOR']==1)).values == False
        
        self.cond1 = np.where(self.cccc==True)
        self.cond2 = np.where(self.cccc==False)
        
        # initializing GP kernel
        self.krn = RBF(length_scale=0.15)
        
        # calculate SNe separation
        self.dist = np.zeros((len(self.z),len(self.z)))
        for j in range(len(self.z)):
            for i in range(len(self.z)):
                self.dist[i,j] = np.abs(self.z[j]-self.z[i])
        
        # applying redshift limits to datapoints, covariance and distance matrices 
        self.cov = np.delete(self.cov,self.cond1[0],1)
        self.cov = np.delete(self.cov,self.cond1[0],0)
        self.dist = np.delete(self.dist,self.cond1[0],1)
        self.dist = np.delete(self.dist,self.cond1[0],0)
        self.z = self.z[self.cond2]
        self.mb = self.mb[self.cond2]
        self.zhel = self.zhel[self.cond2]
         
        # defining positions of calibration SNe and distances to them
        self.sel_cal = b['IS_CALIBRATOR'].values[self.cond2]==1
        self.ceph_dist = b['CEPH_DIST'].values[self.cond2]

    def get_requirements(self):
        return {}
    def logp(self, **params_values):
        
        # getting current parameter from sampler
        H0gp = params_values['H0_gp']
        omegamgp = params_values['omegam_gp']
        Mgp = params_values['M_gp']
        dgp = params_values['d_gp']
        sigmagp = params_values['sigma_gp']
        
        # calculating cosmological distances
        moduli = []
        for  i in range(len(self.z)):
            moduli.append(da(self.z[i], H0gp,omegamgp) * (1. + self.z[i])*(1. + self.zhel[i]))
        
        moduli = np.array(moduli)
        moduli = 5 * np.log10(moduli) + 25

        # replacing distances to calibration SNe with distance measurements
        moduli[self.sel_cal] = self.ceph_dist[self.sel_cal]
        
        # calculating additional covariance matrics
        self.krn.length_scale = dgp
        dd= self.krn(self.dist)
        cov =self.cov.copy() + sigmagp**2 * dd
        
        mb = self.mb
        residuals = mb - Mgp
        residuals -= moduli

        cov = la.cholesky(cov.copy(), lower=True, overwrite_a=True)
        residuals = la.solve_triangular(self.cov.copy(), residuals, lower=True, check_finite=False)
     
        chi2 = (residuals**2).sum()
        return -0.5 * chi2
