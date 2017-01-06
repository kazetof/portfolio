#!/usr/bin/env python
import portfoliofuncs as pf
import numpy as np

data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")
emp_roling_dict = pf.roling_portfolio(data,r0=0.01,window_size=110,methods='empirical',inportfolio_thre=0.01)
lasso_roling_dict = pf.roling_portfolio(data,r0=0.01,window_size=110,methods='lasso',rho=0.4,\
                       		           inportfolio_thre=0.01,using_sklearn=True)
pf.plot_test_return(emp_roling_dict['test_return_array'], lasso_roling_dict['test_return_array'])
pf.plot_abs_change(emp_roling_dict['sol_output_array'], lasso_roling_dict['sol_output_array'])

