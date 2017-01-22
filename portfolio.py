#!/usr/bin/env python
import portfoliofuncs as pf
import numpy as np

data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")

window_size = 110
r0=0.01
inportfolio_thre = 0.01

emp_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='empirical',inportfolio_thre=inportfolio_thre)
emp_diag_roling_dict = pf.roling_portfolio(data,r0=r0,window_size=window_size,\
										methods='empirical_diag',inportfolio_thre=inportfolio_thre)
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
equal_roling_dict = pf.equal_rolling_portfolio(data, window_size=110)

pf.evaluation(emp_roling_dict)
pf.evaluation(emp_diag_roling_dict)
pf.evaluation(lasso_roling_dict)
pf.evaluation(lasso_optim_roling_dict)
pf.evaluation(shrunk_roling_dict)
pf.evaluation(sindex_roling_dict)
pf.evaluation(equal_roling_dict)

pf.plot_test_return(emp_roling_dict, lasso_optim_roling_dict, emp_diag_roling_dict)
pf.plot_turnover(emp_roling_dict, lasso_optim_roling_dict, emp_diag_roling_dict)
pf.plot_abs_change(emp_roling_dict, lasso_optim_roling_dict, emp_diag_roling_dict)
pf.plot_stock_num(0.01, emp_roling_dict, lasso_optim_roling_dict, emp_diag_roling_dict)

pf.plot_test_return(sindex_roling_dict, lasso_optim_roling_dict,equal_roling_dict)
pf.plot_turnover(sindex_roling_dict, lasso_optim_roling_dict)
pf.plot_abs_change(sindex_roling_dict, lasso_optim_roling_dict)
pf.plot_stock_num(0.01, sindex_roling_dict, lasso_optim_roling_dict)

pf.plot_efficient_frontier(data)


"""
lasso_optim_dict, lasso_param_range = pf.lasso_param_optim(data)
#pf.save_dic(lasso_optim_dict, "/Users/kazeto/Desktop/GradThesis/nikkei/output/lasso_optim_dict_48range.pkl")
pf.plot_lasso_param_optim_return(lasso_optim_dict, lasso_param_range)
pf.plot_lasso_param_optim_risk(lasso_optim_dict, lasso_param_range)

#####
##single index
#market portfolio
S_single_index_diag = pf.make_single_index_diagonal_covariance_matrix(data)
S_single_index = pf.make_single_index_covariance_matrix(data)

ticker_dic = { 1801:"大成建設",1803:"清水建設",1963:"日揮",2502:"アサヒグループホールディングス",2801:"キッコーマン",\
            2802:"味の素",2914:"日本たばこ産業",4021:"日産化学工業",4151:"協和発酵キリン",4452:"花王",4503:"アステラス製薬",4507:"塩野義製薬",\
            4543:"テルモ",5101:"横浜ゴム",5108:"ブリヂストン",5406:"神戸製鋼所",5541:"大平洋金属",5631:"日本製鋼所",5713:"",6301:"コマツ",\
            6302:"住友重機械工業",6326:"クボタ",6305:"日立建機",6366:"千代田化工建設",6367:"ダイキン工業",6479:"ミネベア",6770:"アルプス電気",6773:"パイオニア",\
            6954:"ファナック",7202:"いすゞ自動車",7270:"富士重工業",8001:"",8002:"丸紅",8015:"豊田通商",8028:"ユニー・ファミリーマートホールディングス",\
            8830:"住友不動産",9007:"小田急電鉄",9009:"京成電鉄",9433:"KDDI",9503:"関西電力",9531:"東京ガス",9983:"ファーストリテイリング",\
            9984:"ソフトバンクグループ" }

import pandas as pd
names = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/from2000name.csv")
m = data.mean(0)
std = data.std(0)
covmat = np.cov(data, rowvar=0, bias=0)
flontier = pd.DataFrame(efficient_frontier(data, m, covmat),columns=["mean","std"])

df = pd.DataFrame(data=np.c_[m,std],index=np.arange(len(m)),columns=["mean","std"])
df_mean_sorted = df.sort(columns="mean",ascending=False)
df_std_sorted = df.sort(columns="std",ascending=True)
df_rank_mean = df_mean_sorted.index
df_rank_std = df_std_sorted.index
#m_sorted = np.sort(m)[::-1]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x=df["std"],y=df["mean"])
plt.xlabel("std")
plt.ylabel("mean")
plt.title("Mean - Standard Deviation")
#rank = df_rank_mean[0] #The top of mean index.
#plt.text(x=df.loc[rank]['std'], y=df.loc[rank]['mean'], s=str(names[rank]), ha = 'center', va = 'bottom') 
ax.plot(flontier['std'],flontier['mean'],c="r")
fig.show()

for rank in df_rank_mean[0:5]:
	#np.sort(m)[::-1][rank-1]
	print("mean : {}".format(df['mean'][rank]))
	rank_ticker = names[rank]
	print(rank_ticker)
	try:
		rank_name = ticker_dic[int(rank_ticker)]
	except KeyError:
		rank_name = "None in dict"

	print(rank_name)
	text(, 1, rank_name, ha = 'center', va = 'bottom') 

r0_list = np.linspace(0.005,0.01,num=100)


for i in r0_list[0:1]:
	sol,x = pf.mean_variance_model_optim(data,r=None,S=None,r0=i)
	weight = np.array(x).flatten()
	p_mean = np.dot(weight.T,m)
	p_std = np.sqrt(np.dot(np.dot(weight.T,covmat),weight))
"""


