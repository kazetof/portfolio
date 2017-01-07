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
lasso_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,methods='lasso',rho=0.4,\
                       		           inportfolio_thre=inportfolio_thre,using_sklearn_glasso=True)
shrunk_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='shrunk',inportfolio_thre=inportfolio_thre)


def shrunk_param_optim(data):
	shrunk_param_range = np.arange(0.1,1,0.1)
	shrunk_optim_dict = {}
	for i in shrunk_param_range:
		shrunk_optim_dict[str(i)] = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='shrunk',inportfolio_thre=inportfolio_thre, shrunk_param=i)
	for i in shrunk_param_range:
		print(i)
		print(shrunk_optim_dict[str(i)]['expected_return'])
		print(shrunk_optim_dict[str(i)]['risk'])
		#pf.plot_turnover(shrunk_optim_dict[str(i)])
		#pf.plot_abs_change(shrunk_optim_dict[str(i)])


pf.evaluation(emp_roling_dict)
pf.evaluation(iso_roling_dict)
pf.evaluation(lasso_roling_dict)
pf.evaluation(shrunk_roling_dict)

pf.plot_test_return(emp_roling_dict, iso_roling_dict, lasso_roling_dict, shrunk_roling_dict)
pf.plot_turnover(emp_roling_dict, iso_roling_dict, lasso_roling_dict, shrunk_roling_dict)
pf.plot_abs_change(emp_roling_dict, iso_roling_dict, lasso_roling_dict, shrunk_roling_dict)



###
import sklearn.covariance as cov
model = cov.LedoitWolf(assume_centered=False)
model.fit(data)
precision = model.get_precision()
S = np.linalg.inv(precision)
pf.heatmap(np.cov(data.T))
pf.heatmap(S)

##single index
#market portfolio
equal_weight = np.ones(data.shape[1]) / data.shape[1]
market_return = np.array([np.dot(data,equal_weight)]).T
from sklearn import linear_model
model = linear_model.LinearRegression()
indiv_stock = np.array([data[:,0]]).T
model.fit(market_return, indiv_stock)
plot(market_return)
plot(indiv_stock)
print(model.intercept_)
print(model.coef_)
beta = model.coef_[0][0]
resid = indiv_stock - model.predict(indiv_stock)
plot(resid)
single_index_variance = beta**2 * np.var(market_return) + np.var(resid)
emp_variance = np.var(indiv_stock)
print(single_index_variance)
print(emp_variance)



