from cobaya.likelihood import Likelihood
import numpy as np
import scipy.linalg as la
from sklearn.gaussian_process.kernels import RBF
import astropy.constants
import scipy.integrate as intgr
import pandas as pn

class panp_gp_RBF_det(Likelihood):
    def initialize(self):
        
        b=pn.read_csv('/net/comas/data/users/bidenko/cobaya/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat',delimiter = ' ')
        self.cov = np.reshape(pn.read_table(r'/net/comas/data/users/bidenko/cobaya/DataRelease/Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES_STAT+SYS.cov',sep=' ').values,(1701,1701))

        self.z = b['zHD'].values

        
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



    def get_requirements(self):
        return {}
    def logp(self, **params_values):
      
        dgp = params_values['d_gp']
        sigmagp = params_values['sigma_gp']

        self.krn.length_scale = dgp
        dd= self.krn(self.dist)
        det =  np.linalg.slogdet(self.cov.copy() + sigmagp**2 * dd)[1]

        return - 1./2. * det
