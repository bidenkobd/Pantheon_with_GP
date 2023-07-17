### Data
Data used in this work is in `data` folder

Nominal SN and Cepheid-host Distances used in Pantheon+ arXiv:2202.04077 and SH0ES arXiv:2112.04510 are provided in:
Pantheon+SH0ES.dat


Stat+Systematic covariance matrix is provided in:

Pantheon+SH0ES_STAT+SYS.cov


Data from the Pantheon analysis in arxiv:1710.00845 provided in:

lcparam_def.txt
lcparam_DS17f.txt.


Stat+Systematic covariance matrix is provided in:

sys_full_long.txt


Please, remember to cite original papers.

### Likelihoods
Likelihoods for reproducing results with Cobaya framework are in `likelihoods`.
Recommended usage: 
   .. code:: bash
   
      $ mpirun -n [n_processes] cobaya-run [filename].yaml


### Parameter constraints
MCMC cahins presented in results section of this work and appendix are in `chains`

### Data analysis and scripts for figures
Jupyter notebooks with scrits to generate figures from the paper are in `notebooks`

