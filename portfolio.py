#!/usr/bin/env python
import portfoliofuncs as pf
import numpy as np

data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")
window_size = 110
r0=0.01
inportfolio_thre = 0.01
emp_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='empirical',inportfolio_thre=inportfolio_thre)
lasso_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,methods='lasso',rho=0.4,\
                       		           inportfolio_thre=inportfolio_thre,using_sklearn=True)
shrunk_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='shrunk',inportfolio_thre=inportfolio_thre)

pf.plot_test_return(emp_roling_dict, lasso_roling_dict, shrunk_roling_dict)
pf.turnover_plot(emp_roling_dict['sol_output_array'], lasso_roling_dict['sol_output_array'])
pf.plot_abs_change(emp_roling_dict['sol_output_array'], lasso_roling_dict['sol_output_array'])


import sklearn.covariance as cov
model = cov.LedoitWolf(assume_centered=False)
model.fit(data)
precision = model.get_precision()
S = np.linalg.inv(precision)
pf.heatmap(np.cov(data.T))
pf.heatmap(S)

model = cov.ShrunkCovariance(shrinkage=0.9,assume_centered=False)
model.fit(data)
precision = model.get_precision()
S2 = np.linalg.inv(precision)
pf.heatmap(S2)
