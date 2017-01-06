import numpy as np
import pandas as pd
########
#Convert to excel to format which we can compute.
########
name = np.loadtxt('/Users/kazeto/Desktop/GradThesis/nikkei/names.csv',delimiter=',').astype(int)

#data.csv is the data wihch is just converted from original xlsx to csv.
d = pd.read_csv("/Users/kazeto/Desktop/GradThesis/nikkei/data.csv", header=None)
#We use pandas because pansas will fill the blanks by nan.
d = np.array(d)

cnum = 4 #the number of columns which each ticker has.
nikkei_dict = {}
for i in np.arange(len(name)):
    nikkei_dict[name[i]] = d[3:,i*cnum:i*cnum+cnum]
    #Cutting the 'NAME?' error by starting from 3. The error is because of remaining the blomberg API.

times = np.array([nikkei_dict[n][0][0] for n in name])
from2000index = np.where(times == '2000/2/29')
from2000keys = name[from2000index].astype(int)

data = np.array([nikkei_dict[n][:,2] for n in from2000keys]).T
#2 is PX_LAST data.
data = data.astype(float)

np.savetxt("/Users/kazeto/Desktop/GradThesis/nikkei/from2000data.csv", data, delimiter=",")
np.savetxt("/Users/kazeto/Desktop/GradThesis/nikkei/from2000name.csv", from2000keys, delimiter=",")

######
#convert to price to return ratio.
######

def logdiff(x):
    x = log(x)
    x1 = list(np.r_[x,0])
    del x1[0]
    x1 = np.array(x1)
    x2 = list(x1 - x)
    N = len(x2)
    del x2[N-1]
    x2 = np.array(x2)
    return x2

data2 = np.array([logdiff(data[:,i]) for i in np.arange(data.shape[1])]).T
np.savetxt("logdiffdata.csv", data2, delimiter=",")

############
#TOPIX core30
############
name = np.loadtxt('./names.csv',delimiter=',').astype(int)
core30 = [2914,3382,4063,4502,4503,6501,6752,6758,6861,6902,6954,6981,7201,7203,7267,7751,8031,8058,8306,8316,8411,8766,8801,8802,9020,9022,9432,9433,9437,9984]
core30_droped = [2914,3382,4063,4502,4503,6501,6752,6758,6902,6954,7201,7203,7267,7751,8031,8058,8306,8316,8411,8766,8801,8802,9020,9022,9432,9433,9437,9984]

core30_dict = {}
for i in np.arange(len(core30_droped)):
    core30_dict[core30_droped[i]] = nikkei_dict[core30_droped[i]]
core30_times = np.array([core30_dict[n][0][0] for n in core30_droped])
core30_from2000index = np.where(core30_times == '2000/2/29')[0]
core30_from2000keys = np.array(core30_droped)[core30_from2000index]
core30_data = np.array([core30_dict[n][:,2] for n in core30_from2000keys]).T
core30_data = core30_data.astype(float)
core30_data2 = np.array([logdiff(core30_data[:,i]) for i in np.arange(core30_data.shape[1])]).T
np.savetxt("/Users/kazeto/Desktop/nikkei/core30_logdiffdata.csv", core30_data2, delimiter=",")
