# based on data from Riess 2016
from cobaya.likelihood import Likelihood
import numpy as np

class pan_sh0es_M_calib(Likelihood):
    def initialize(self):
        self.Mb_mean = np.array([-19.389, -19.047, -19.331, -19.39 , -19.111, -19.236, -19.535,
          -19.161, -19.207, -19.103, -19.507, -19.058, -19.534, -19.293,
          -19.113, -19.085, -19.255, -19.196, -19.449], 'float64')
        self.eMb = np.array([0.125, 0.147, 0.128, 0.137, 0.125, 0.152, 0.147, 0.125, 0.131,
          0.136, 0.134, 0.16 , 0.311, 0.135, 0.135, 0.124, 0.154, 0.139,
          0.13 ], 'float64')

    def get_requirements(self):
        return {}
    def logp(self, **params_values):

        Mgp = params_values['M_gp']

        return -0.5 * sum((self.Mb_mean - Mgp)**2/self.eMb**2)
 