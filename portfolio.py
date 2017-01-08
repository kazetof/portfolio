#!/usr/bin/env python
import portfoliofuncs as pf
import numpy as np

data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")

window_size = 110
r0=0.01
inportfolio_thre = 0.01

emp_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='empirical',inportfolio_thre=inportfolio_thre)
iso_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='empirical_isotropy',inportfolio_thre=inportfolio_thre)
lasso_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,methods='lasso',\
                       		           inportfolio_thre=inportfolio_thre,using_sklearn_glasso=True,\
                       		           shrunk_param=0.01)
lasso_optim_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,methods='lasso',\
                       		           inportfolio_thre=inportfolio_thre,using_sklearn_glasso=True,\
                       		           shrunk_param=0.0035)
shrunk_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='shrunk',inportfolio_thre=inportfolio_thre)
sindex_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='singleindex',inportfolio_thre=inportfolio_thre)

pf.evaluation(emp_roling_dict)
pf.evaluation(iso_roling_dict)
pf.evaluation(lasso_roling_dict)
pf.evaluation(lasso_optim_roling_dict)
pf.evaluation(shrunk_roling_dict)
pf.evaluation(sindex_roling_dict)

pf.plot_test_return(emp_roling_dict, lasso_optim_roling_dict, iso_roling_dict)
pf.plot_turnover(emp_roling_dict, lasso_optim_roling_dict, iso_roling_dict)
pf.plot_abs_change(emp_roling_dict, lasso_optim_roling_dict, iso_roling_dict)
pf.plot_stock_num(0.01, emp_roling_dict, lasso_optim_roling_dict, iso_roling_dict)

plt.plot(emp_roling_dict['d_window_mean'])
plt.plot(emp_roling_dict['d_window_variance'])

lasso_optim_dict, lasso_param_range = pf.lasso_param_optim(data)
#pf.save_dic(lasso_optim_dict, "/Users/kazeto/Desktop/GradThesis/nikkei/output/lasso_optim_dict_48range.pkl")
pf.plot_lasso_param_optim_return(lasso_optim_dict, lasso_param_range)
pf.plot_lasso_param_optim_risk(lasso_optim_dict, lasso_param_range)

#####
import sklearn.covariance as cov
model = cov.LedoitWolf(assume_centered=False)
model.fit(data)
precision = model.get_precision()
S = np.linalg.inv(precision)
pf.heatmap(np.cov(data.T))
pf.heatmap(S)

##single index
#market portfolio
S_single_index_diag = pf.make_single_index_diagonal_covariance_matrix(data)
S_single_index = pf.make_single_index_covariance_matrix(data)

