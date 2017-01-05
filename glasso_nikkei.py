#!/usr/bin/env python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx

def logdiff(x):
    """
        input paramater
        ----------------------
        x : ndarray
            (n*1) vector
        ---------------------

        returns
        ---------------------
        x2 : ndarray
            (n-1)*1 vector
            taken log and one diff.
        ---------------------

    """
    x = np.log(x)
    x1 = list(np.r_[x,0])
    del x1[0]
    x1 = np.array(x1)
    x2 = list(x1 - x)
    N = len(x2)
    del x2[N-1]
    x2 = np.array(x2)
    return x2

def cofactor_matrix(X,num):
    """
        input paramater
        ----------------------
        X : ndarray
            (p*p) matrix
        num : integer
            cofactor index
        ---------------------

        returns
        ---------------------
        output_X : (p-1)*(p-1) matrix ommiting num column and row.
        output_Y : (p*1) vector num column
        ---------------------

    """
    M = np.shape(X)[1]
    index = np.ones(M, dtype=bool)
    index[num] = False
    X_2 = X[index]
    X_3 = X_2.T[index]
    output_X = X_3.T

    Y_1 = X[:,num]
    output_Y = np.delete(Y_1,num)
    return output_X,output_Y

def soft_thre(x,lam):
    """
        input paramater
        ----------------------
        x : float
        lam : float
            threshold
        ---------------------

        returns
        ---------------------
        s : float
        ---------------------

    """
    if x > 0 and lam < np.abs(x):
        s = x - lam
    elif x < 0 and lam < np.abs(x):
        s = x + lam
    else:
        s = 0

    return s

def inf_debug(matrix,name):
    """
        input paramater
        ----------------------
        matrix : ndarray
            (p*p) matrix
            esitmating covariance matrix.
        ---------------------

        returns
        ---------------------
        None
            if matrix contains inf then alert it.
        ---------------------

    """
    if np.any(np.isinf(matrix)):
        print("!!!!!!!!!!!!!! inf in {} !!!!!!!!!!!!!!!!".format(name))
        print(np.where(matrix==np.float(inf)))
        print(matrix[np.where(matrix==np.float(inf))])


def update_W12(W,S,num,rho,lam):
    """
        input paramater
        ----------------------
        W : ndarray
            updating covariance matrix.
        S : ndarray
            empirical covariance matrix.
        num : integer
            iterating index (1~p)
        rho :
        lam :
        ---------------------

        returns
        ---------------------
        W_new : ndarray
        ---------------------

    """
    D = np.shape(W)[1]

    W_11,W_12 = cofactor_matrix(W,num)
    S_11,S_12 = cofactor_matrix(S,num)

    l,P = np.linalg.eig(W_11)
    L = np.diag(np.sqrt(l))
    W_11_sqrt = np.dot(np.dot(P,L),np.linalg.inv(P))
    #np.dot(P.T,P)
    #W_check = np.dot(W_11_sqrt,W_11_sqrt)
    #W_11 - W_check

    b = np.dot(np.linalg.inv(W_11_sqrt),S_12)

    D2 = np.shape(W_11)[1]

    beta_old = np.zeros(D2)
    beta_new = np.copy(beta_old)

    #CD
    for k in np.arange(D2):
        index = np.ones(D2, dtype=bool)
        index[k] = False

        W_kj,no_use = cofactor_matrix(W_11,k)

        term1 = S_12[k] - sum(np.dot(W_kj,beta_old[index]))
        term2 = 1. / W_11[k][k]
        #print 'term2:',term2 #debug
        beta_new[k] = soft_thre(term1,lam) * term2
        #print 'beta_new',beta_new[k] #debug
        beta_old = np.copy(beta_new)
    #print(beta_new)

    W_12_new = np.dot(W_11,beta_new)
    W_newcolum = np.insert(W_12_new,[num],np.diag(W)[num])
    W_new = np.copy(W)
    W_new[:,num] = W_newcolum
    W_new[num,:] = W_newcolum
    #print(W_new)
    return W_new


def cov_lasso_optim(S,N,M,rho,lam_rho_ratio=0.08):
    """
        input paramater
        ----------------------
        S : ndarray
            (p*p) matrix
        N : integer
            number of rows
        M : integer
            number of columns
        rho : integer
            parameter of adding value to diagonal elements in S.
        lam_rho_ratio :
            parameter of regularization in each iterate step.
        ---------------------

        returns
        ---------------------
        W_new : ndarray
            estimated covariance matrix
            (p*p) matrix
        ---------------------

    """
    W = S + np.diag(np.tile(rho,M))
    for j in np.arange(1):
        for i in np.arange(M):
            if i == 0 and j == 0:
                W_old = np.copy(W)

            W_new = update_W12(W_old,S,i,rho,lam=rho*lam_rho_ratio) #rho=0.1,lam=rho*0.08 looks good
            #If we choose under rho=0.1,lam=rho*0.074, W_new includes inf
            W_old = np.copy(W_new)
    #print "W_new : ",W_new
    return W_new


#this function return just 0 or 1 matrix
def get_edge_matrix(W):
    """
        input paramater
        ----------------------
        W : ndarray
            (p*p) matrix
            estimated covariance matrix
        ---------------------

        returns
        ---------------------
        edge : ndarray
            (p*p) matrix
            nonzero elements was replaced by 1.
            The edge elements(edge_ij) is 0 or 1.
        ---------------------

    """
    M = np.shape(W)[1]
    i_index, j_index = np.nonzero(W)
    edge = np.zeros((M,M))
    for (i,j) in zip(i_index,j_index):
        edge[i][j] = 1

    return edge

def plot_graph(W,color,name=None):
    """
        input paramater
        ----------------------
        W : ndarray
            estimated covariance matrtix
        color : string
            node color in plot.
            ex. 'r' is red.
        ---------------------

        returns
        ---------------------
        None
            plotting network graph.
        ---------------------

    """
    plt.figure()
    D = np.shape(W)[1]
    G = nx.Graph()
    for i in np.arange(D):
        G.add_node(i)

    i_index, j_index = np.nonzero(W)
    for (i,j) in zip(i_index,j_index):
        G.add_edge(i, j)

    labels={}
    for i in np.arange(D):
        labels[i] = str(i)

    pos=nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos,node_color=color)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos,labels,font_size=16)
    #title(name)

def cor_mat(S):
    """
        input paramater
        ----------------------
        S : ndarray
            (p*p) matrix
            covariance matrix
        ---------------------

        returns
        ---------------------
        cor_matrix : ndarray
            (p*p) matrix
            correlation matrix
        ---------------------

    """
    std_array = np.array(np.sqrt([np.diag(S)])).T
    s = np.dot(std_array,std_array.T) #norm
    cor_matrix = S/s
    return cor_matrix

def heatmap(matrix,title=None):
    """
        input paramater
        ----------------------
        matrix : ndarray
            (p*p) matrix
        title : string
            title of plot
        ---------------------

        returns
        ---------------------
        None
            plotting heat map of covariance matrix.
        ---------------------

    """
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    X,Y = np.meshgrid(x,y)
    fig, ax = plt.subplots()
    ax.pcolor(X,Y,matrix)
    plt.title(title)
    #ax.pcolor(X,Y,matrix, cmap=plt.cm.Blues)
    fig.show()

def extract_nonzero(matrix):
    #This function return the submatrix which the non diagonal elements are not zero.
    """
        input paramater
        ----------------------
        matrix : ndarray
            (p*p) matrix
        ---------------------

        returns
        ---------------------
        nonzeroindex : ndarray
            m*1 vector
            ( m is number of column which has nonzero value
            in non diagonal elements. )
        output : ndarray
            (m*m) matrix
        ---------------------

    """
    nonzeroindex = np.where(np.sum(matrix,0)-np.diag(matrix) > 0)
    temp = matrix[nonzeroindex]
    output = temp.T[nonzeroindex]
    return nonzeroindex,output

############
#portfolio optimization
############
import cvxopt
from cvxopt import solvers,sparse,printing
def mean_variance_model_optim(data,r=None,S=None,r0=0.01):
    """
        input paramater
        ----------------------
        data : ndarray
            (n*p) matrix
        r : ndarray
            mean vector
            (p*1) vector
        S : ndarray
            covariance matrix
            (p*p) matrix
        r0 : float
            lower bound of expected return.

        If r and S are None, then caluculate empirical mean
        and covariance matrix.
        ---------------------

        returns
        ---------------------
        sol : dictinary
            solution of quadratic programming.
        x : ndarray
            (p*1) vector
            the weight of portfolio
        ---------------------

    """
    N = data.shape[0]
    if r == None:
        r = np.mean(data,0)
    if S == None:
        diff = data - r
        S = (1/float(N)) * np.dot(diff.T,diff)

    minus_r = np.matrix(-np.copy(r))
    p = data.shape[1]

    P = cvxopt.matrix(np.copy(S))
    q = cvxopt.matrix(0.0,(p,1))
    I = cvxopt.matrix(0.0,(p,p))
    I[::p+1] = -1.0
    G = sparse([I])
    A = sparse([cvxopt.matrix(minus_r),cvxopt.matrix(1.0,(1,p))])
    b = cvxopt.matrix([-r0,1])
    h = cvxopt.matrix(np.zeros(p))
    sol = solvers.qp(P,q,G,h,A,b)
    x = sol['x']
    print("ratio of portfolio : {}".format(x))
    print("portfolio return os {}".format(np.sum(cvxopt.mul(x,cvxopt.matrix(r)))))
    print("sum of ratio x is {}".format(np.sum(x)))
    return sol,x

def split_data(d,split_t):
    """
        input paramater
        ----------------------
        d : ndarray
         (n*p) matrix
        split_t : integer
            split data into training data and test data
            in index of split_t.
        ---------------------

        returns
        ---------------------
        d1 : ndarray
            traing data
        d2 : ndarray
            test data
        ---------------------

    """
    d1 = d[0:split_t,:]
    d2 = d[split_t:,:]
    print("d1.shape : ",d1.shape)
    print("d2.shape : ",d2.shape)
    return(d1,d2)

def window_data(d,start,window_size=100):
    """
        input paramater
        ----------------------
        d : ndarray
            (n*p) matrix
        start : integer
            index where window starts
        window_size : integer
            output window data contains in this number of data.
        ---------------------

        returns
        ---------------------
        window_d : ndarray
        ---------------------

    """
    window_d = d[start:start+window_size,:]
    return window_d

def roling_portfolio(d,r0=0.01,window_size=100,rho=0.4,lam_rho_ratio=0.2,\
                    inportfolio_thre=0.01, norm_bool=True):
    """
        input paramater
        ----------------------
        d : ndarray
        r0 : float
            expecting return which the portfolio must satisfy.
        window_size : integer
            the range of window which caluculate ratio of portfolio.
        rho : float
            adding value to diagonal element in coordinate descent algorithm.
        lam_rho_ratio : float
            descide lambda in this regularlization
            (lambda is parameter of strongthness of regularlization)
        ---------------------

        returns
        ---------------------
        back_up_dict : dictionary
            test_retrun_emp_array ; return in test data for portfolio which used empirical covariance matrix.
            test_return_lasso_array ; return in test data for portfolio which used lasso covariance matrix.
        ---------------------

    """
    #define list to save
    test_retrun_emp_array = []
    test_return_lasso_array = []
    emp_true_variance_array = []
    lasso_true_variance_array = []
    sol_emp_output_array = []
    sol_lasso_output_array = []
    emp_status_array = []
    lasso_status_array = []
    is_in_emp_portfolio_array = []
    is_in_lasso_portfolio_array = []

    cvxopt.matrix_repr = printing.matrix_str_default #for dealing cvxopt matrix as np_matrix.
    import sklearn.covariance as cov
    model = cov.GraphLasso(alpha=0.01,mode='cd',tol=1e-3)

    for start in np.arange(len(d) - window_size -1):
        print("----------- step : {} -----------".format(start))
        d_window = window_data(d,start,window_size)
        S_window = np.dot(d_window.T,d_window) / d_window.shape[0]
        N_window = d_window.shape[0]
        M_window = d_window.shape[1]
        #W_window = cov_lasso_optim(S=S_window,N=N_window,M=M_window,rho=rho,lam_rho_ratio=lam_rho_ratio)
        model.fit(d_window)#for debug
        W_window = model.covariance_#for debug
        sol_empirical,r1 = mean_variance_model_optim(d_window,r0=r0)
        sol_lasso,r2 = mean_variance_model_optim(d_window,S=W_window,r0=r0)
        sol_emp_output = sol_empirical['x']
        sol_lasso_output = sol_lasso['x']
        is_in_emp_portfolio = np.asarray(sol_emp_output).flatten() > inportfolio_thre
        is_in_lasso_portfolio = np.asarray(sol_lasso_output).flatten() > inportfolio_thre
        testdata = d[start+window_size+1:,:]

        if norm_bool == True:
            sol_emp_norm_output = normalized_ratio(sol_emp_output,thre=inportfolio_thre)
            sol_lasso_norm_output = normalized_ratio(sol_lasso_output,thre=inportfolio_thre)
            test_retrun_emp = np.dot(testdata,sol_emp_norm_output)[0]
            test_return_lasso = np.dot(testdata,sol_lasso_norm_output)[0]
        else:
            test_retrun_emp = np.dot(testdata,sol_emp_output)[0]
            test_return_lasso = np.dot(testdata,sol_lasso_output)[0]

        test_retrun_emp_array.append(test_retrun_emp)
        test_return_lasso_array.append(test_return_lasso)

        emp_status_array.append(sol_empirical['status'])
        lasso_status_array.append(sol_lasso['status'])

        sol_emp_output_list = np.asarray(sol_emp_output).flatten()
        sol_lasso_output_list = np.asarray(sol_lasso_output).flatten()
        sol_emp_output_array.append(np.array(sol_emp_output_list))
        sol_lasso_output_array.append(np.array(sol_lasso_output_list))
        is_in_emp_portfolio_array.append(is_in_emp_portfolio)
        is_in_lasso_portfolio_array.append(is_in_lasso_portfolio)

        #calculate true(base) variance.
        emp_true_variance = np.std(np.dot(d[start + window_size:,:],sol_emp_output))
        lasso_true_variance = np.std(np.dot(d[start + window_size:,:],sol_lasso_output))
        emp_true_variance_array.append(emp_true_variance)
        lasso_true_variance_array.append(lasso_true_variance)
        print("N,M : ",N_window,M_window)
        print("Empirical Optimal Solution : {} , LASSO Optimal Solution : {}".format(sol_empirical['status'],sol_lasso['status']))
        #print "S : ",S_window
        #print "sol_emp_output : ",np.array(sol_emp_output)
        #print "sol_lasso_output : ",np.array(sol_lasso_output)

    emp_diff = np.array(emp_true_variance_array) - np.array(test_retrun_emp_array)
    lasso_diff = np.array(lasso_true_variance_array) - np.array(test_return_lasso_array)

    #save to dict
    back_up_dict = {}
    back_up_dict['test_retrun_emp_array'] = np.array(test_retrun_emp_array).flatten()
    back_up_dict['test_return_lasso_array'] = np.array(test_return_lasso_array).flatten()
    back_up_dict['expected_return_emp'] = np.mean(test_retrun_emp_array) * 12
    back_up_dict['expected_return_lasso'] = np.mean(test_return_lasso_array) * 12
    back_up_dict['risk_emp'] = np.std(test_retrun_emp_array) * 12
    back_up_dict['risk_lasso'] = np.std(test_return_lasso_array) * 12
    back_up_dict['emp_true_variance_array'] = np.array(emp_true_variance_array)
    back_up_dict['lasso_true_variance_array'] = np.array(lasso_true_variance_array)
    back_up_dict['emp_diff'] = np.array(emp_diff)
    back_up_dict['lasso_diff'] = np.array(lasso_diff)
    back_up_dict['mean_emp_diff'] = np.mean(emp_diff)
    back_up_dict['mean_lasso_diff'] = np.mean(lasso_diff)
    back_up_dict['sol_emp_output_array'] = np.array(sol_emp_output_array)
    back_up_dict['sol_lasso_output_array'] = np.array(sol_lasso_output_array)
    back_up_dict['emp_status_array'] = np.array(emp_status_array)
    back_up_dict['lasso_status_array'] = np.array(lasso_status_array)
    back_up_dict['rho'] = rho
    back_up_dict['lam_rho_ratio'] = lam_rho_ratio
    back_up_dict['window_size'] = window_size
    back_up_dict['r0'] = r0
    back_up_dict['M'] = d.shape[1]
    back_up_dict['is_in_emp_portfolio_array'] = np.array(is_in_emp_portfolio_array)
    back_up_dict['is_in_lasso_portfolio_array'] = np.array(is_in_lasso_portfolio_array)

    if 'unknown' in emp_status_array or 'unknown' in lasso_status_array:
        print("!!!!!!!!Optimal solution was not found!!!!!!!!!")
    else:
        print("Optimal solution was found!")

    return back_up_dict

def check_turnover(output_array,thre=0.01,num_return=True):
    """
        input paramater
        ----------------------
        output_array : ndarray
        ratio of portfolio which is output of quadratic programming.
        thre : float
        If ratio of portfolio is over thre, the stock will be bought.
        If ratio of portfolio is under thre, the stock will be sold.
        ---------------------

        returns
        ---------------------
        num_turnover : integer
        number of buy and sell
        ---------------------

    """
    def check_change(vector):
        bool_turnover = np.array([ vector[i] != vector[i+1] for i in np.arange(len(vector)-1)])
        return np.sum(bool_turnover)

    bool_array = output_array > thre
    turnover_array = [ check_change(bool_array[:,i]) for i in np.arange(bool_array.shape[1]) ]
    num_turnover = np.sum(turnover_array)
    if num_return == True:
        return num_turnover
    else:
        return turnover_array

def turnover_plot(sol_emp_output_array,sol_lasso_output_array):
    range_list = np.arange(0.01,0.5,0.01)
    emp_turnover_each_thre = np.array([ check_turnover(sol_emp_output_array,thre=i,num_return=True) for i in range_list ])
    lasso_turnover_each_thre = np.array([ check_turnover(sol_lasso_output_array,thre=i,num_return=True) for i in range_list ])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range_list,emp_turnover_each_thre,label="Empirical turnover num")
    ax.plot(range_list,lasso_turnover_each_thre,label="Lasso turnover num")
    ax.legend()
    plt.xlabel("threshould of portfolio")
    plt.ylabel("number of turnover")
    plt.title("number of turnover")

def normalized_ratio(ratio_vector,thre=0.01):
    """
        input paramater
        ----------------------
        ratio_vector : ndarray
            n*1 vector
            ratio of portfolio which is output of quadratic programming.
        thre : float
            The thereshuld of portfolio.
            The stocks which is under thre will be 0 and another stocks
            will be normalized to be sum is ten.
        ---------------------

        returns
        ---------------------
        ratio_vector_normalized : ndarray
        n*1 vector
        ---------------------

    """
    if isinstance(ratio_vector, cvxopt.base.matrix) == True:
        ratio_vector = np.array(ratio_vector).flatten()
    input_vector = ratio_vector.copy() #input_vector[bool_vector] = 0.0 will change original vector so copy it.
    bool_vector = input_vector < thre
    input_vector[bool_vector] = 0.0

    if np.sum(input_vector) == 0:
        ratio_vector_normalized = input_vector #avoiding nan
    else:
        ratio_vector_normalized = input_vector / np.sum(input_vector)
    return ratio_vector_normalized

def check_abs_change_portfolio(output_array,vector_return_bool=False):
    """
        input paramater
        ----------------------
        output_array : ndarray
        n*p matrix (n is time and p is number of stocks)
        ratio of portfolio which is output of quadratic programming.

        vector_return_bool : bool
        if vector_return_bool is True, then this function will return
        abs_change_vector. if it is False, then return abs_change.
        ---------------------
        returns
        ---------------------
        abs_change_vector : ndarray
        (p*1) vector
        absolute value of change for each stock.

        abs_change : float
        absolute value of change for all over stocks.
        ---------------------

    """
    A = np.r_[np.array([np.zeros(output_array.shape[1])]), output_array]
    B = np.r_[output_array, np.array([np.zeros(output_array.shape[1])])]
    C = B - A
    D = C[1:C.shape[0]-1,:] #D is (n-1 * p matrix) since taking difference of t and t-1
    abs_change_vector = np.sum(np.abs(D),0)
    abs_change = np.sum(abs_change_vector)
    if vector_return_bool==True:
        return abs_change_vector
    else:
        return abs_change

def plot_test_return(roling_back_up_dict):
    """
        input paramater
        ----------------------
        roling_back_up_dict : dict
        output of roling_portfolio function.
        ---------------------
        returns
        ---------------------
        None
        ---------------------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(roling_back_up_dict['test_retrun_emp_array'], 'b', label="return of Empirical Covariance")
    ax.plot(roling_back_up_dict['test_return_lasso_array'], 'r', label="return of Estimated Covariance by LASSO")
    plt.title("Comparison of Retrun")
    ax.legend(loc = 'bottom left')
    plt.ylabel("return of test data")
    plt.xlabel("time")
    fig.show()


data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")

#data = data - np.mean(data,0)

N = data.shape[0]
M = data.shape[1]
S = np.dot(data.T,data) / N
rho = 0.1

#Caluculate corelation matrix
cor_mat1 = cor_mat(S)
print('correlation',cor_mat1)

W_new = cov_lasso_optim(S=S,rho=rho,N=N,M=M)
import sklearn.covariance as cov
model = cov.GraphLasso(alpha=0.01,mode='cd')
model.fit(data)
cov_ = model.covariance_

edge = get_edge_matrix(W_new)
non = np.nonzero(W_new)
print(edge)

plot_graph(W_new,'r')
cor = cor_mat(W_new)
print('Estimated Covariance',s)
print('cor',cor)

#####
#debuging
#####

"""
#checking each steps for determing rho and lambda.
i=0
rho = 0.1
W = S + np.diag(np.tile(rho,M))
W_old = np.copy(W)
W_new = update_W12(W_old,S,i,rho,lam=rho*0.05)
edge = get_edge_matrix(W_new)
plot_graph(W_new,'r','python')
W_old = np.copy(W_new)

#check where inf appears.
rho = 0.1
W = S + np.diag(np.tile(rho,M))
W_old = np.copy(W)
for i in np.arange(10):
    W_new = update_W12(W_old,S,i,rho,lam=rho*0.05) #rho=0.1,lam=rho*0.05
    W_old = np.copy(W_new)

sum(edge)
"""

#######
#checing the remining relation.
nonzeroindex, nonzeromatrix = extract_nonzero(W_new)
print(nonzeromatrix.shape)
#from2000keys[nonzeroindex] #from2000keys was defined in convert.py
#plot_graph(nonzeromatrix,'r','non zero covariance matrix')
#plot_nonzero = nonzeromatrix/np.max(nonzeromatrix)

#######
#plot heat map
heatmap(S,"S")
heatmap(W_new,"W_new")
heatmap(cov_,"sklearn glasso")
heatmap(nonzeromatrix,"sub-W_new")

######
#comparing to R glasso
for i in np.arange(20):
    Rname ='./W_R/glasso_W_rho' + str(i) + '.csv'
    W_R = np.loadtxt(Rname,delimiter=",",skiprows=1)
    #heatmap(W_R)
    #title('rho=' + str(i))
    nonzeroindex, nonzeromatrix = extract_nonzero(W_R)
    print(len(nonzeroindex[0]))

#This is timeconsuming so It's better to use data[:,0:50] beforehand.
roling_back_up_dict = roling_portfolio(data,r0=0.01,window_size=110,rho=0.4,lam_rho_ratio=0.2,inportfolio_thre=0.01,norm_bool=True)

#roling_back_up_dict = roling_portfolio(data[:,0:50],r0=0.01,window_size=130,rho=0.2,lam_rho_ratio=0.1,norm_bool=True)

dic = {}
for j in np.arange(0.1,1,0.2):
    for i in np.arange(0.1,1.4,0.1):
        roling_back_up_dict = roling_portfolio(data[:,0:50],r0=0.01,window_size=130,rho=i,lam_rho_ratio=j)
        print("rho : {}".format(i))
        print("lam_rho_ratio : {}".format(j))
        print(roling_back_up_dict['emp_status_array'])
        dic[str(i)+str(j)] = roling_back_up_dict['emp_status_array']




plot_test_return(roling_back_up_dict)

print("Empirical Mean : ", np.mean(roling_back_up_dict['test_retrun_emp_array']) * 12)
print("LASSO Mean : ", np.mean(roling_back_up_dict['test_return_lasso_array']) * 12)
print("Empirical Std : ", np.std(roling_back_up_dict['test_retrun_emp_array']) * np.sqrt(12))
print("LASSO Std : ", np.std(roling_back_up_dict['test_return_lasso_array']) * np.sqrt(12))
print("Empirical Diff : ",np.mean(roling_back_up_dict['emp_diff']))
print("LASSO Diff : ",np.mean(roling_back_up_dict['lasso_diff']))

#checking turn over
is_in_emp_portfolio_array = roling_back_up_dict['is_in_emp_portfolio_array']
is_in_lasso_portfolio_array = roling_back_up_dict['is_in_lasso_portfolio_array']

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(np.arange(is_in_emp_portfolio_array.shape[1]),is_in_emp_portfolio_array.astype(int).T)
plt.title("Empirical")
ax2 = fig.add_subplot(122)
ax2.plot(np.arange(is_in_lasso_portfolio_array.shape[1]),is_in_lasso_portfolio_array.astype(int).T)
plt.title("Lasso")
fig.show()

np.sum(is_in_emp_portfolio_array,1)
np.sum(is_in_lasso_portfolio_array,1)
sol_emp_output_array = np.copy(roling_back_up_dict['sol_emp_output_array'])
sol_lasso_output_array = np.copy(roling_back_up_dict['sol_lasso_output_array'])

turnover_plot(sol_emp_output_array,sol_lasso_output_array)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.plot(np.arange(sol_emp_output_array.shape[1]),sol_emp_output_array.T)
plt.title("Empirical")
ax2 = fig.add_subplot(122)
ax2.plot(np.arange(sol_lasso_output_array.shape[1]),sol_lasso_output_array.T)
plt.title("Lasso")
fig.show()

for i in np.arange(50):
    plt.plot(np.arange(sol_emp_output_array.shape[0]),sol_emp_output_array[:,i])

for i in np.arange(50):
    plt.plot(np.arange(sol_lasso_output_array.shape[0]),sol_lasso_output_array[:,i])

emp_turnover = check_turnover(sol_emp_output_array,thre=0.1,num_return=True)
lasso_turnover = check_turnover(sol_lasso_output_array,thre=0.1,num_return=True)
print(emp_turnover)
print(lasso_turnover)

def check_main_stock_in_portfolio(output_array,thre=0.01,):
    bool_array = output_array > thre
    return np.sum(bool_array,0)

def normalized_ratio_array(output_array,thre=0.01):
    output_norm = np.array([ normalized_ratio(output_array[i],thre=thre) for i in np.arange(output_array.shape[0]) ])
    return output_norm

sol_emp_output_array_norm = normalized_ratio_array(sol_emp_output_array,thre=0.01)
check_abs_change_portfolio(sol_emp_output_array_norm)

def plot_abs_change(sol_emp_output_array,sol_lasso_output_array):
    range_list = np.arange(0.01,0.5,0.05)
    emp_change = np.array([ check_abs_change_portfolio(normalized_ratio_array(sol_emp_output_array,thre=i)) for i in range_list ])
    lasso_change = np.array([ check_abs_change_portfolio(normalized_ratio_array(sol_lasso_output_array,thre=i)) for i in range_list ])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(range_list, emp_change, label="Empirical")
    ax1.plot(range_list, lasso_change, label="Lasso")
    plt.title("Abs values of change")
    plt.xlabel("ratio of threshold")
    plt.ylabel("sum of absolute value of change")
    plt.legend()
    fig.show()



[ normalized_ratio_array(sol_emp_output_array,thre=i) for i in np.arange(0.01,0.5,0.05) ]
check_abs_change_portfolio(sol_lasso_output_array)

emp_output_array_norm = normalized_ratio_array(sol_emp_output_array,thre=0.1)
emp_output_array_norm_change = check_abs_change_portfolio(emp_output_array_norm)
print(np.sum(emp_output_array_norm_change))

lasso_output_array_norm = normalized_ratio_array(sol_lasso_output_array,thre=0.1)
lasso_output_array_norm_change = check_abs_change_portfolio(lasso_output_array_norm)
print(np.sum(lasso_output_array_norm_change))

[ check_main_stock_in_portfolio(sol_emp_output_array,thre=i) for i in np.arange(0.01,0.3,0.01) ]
[ check_main_stock_in_portfolio(sol_lasso_output_array,thre=i) for i in np.arange(0.01,0.3,0.01) ]

#stocs name in portfolio
ticker_dic = { 1801:"大成建設",1803:"清水建設",1963:"日揮",2502:"アサヒグループホールディングス",2801:"キッコーマン",\
            2802:"味の素",2914:"日本たばこ産業",4021:"日産化学工業",4151:"協和発酵キリン",4452:"花王",4503:"アステラス製薬",4507:"塩野義製薬",\
            4543:"テルモ",5101:"横浜ゴム",5108:"ブリヂストン",5406:"神戸製鋼所",5541:"大平洋金属",5631:"日本製鋼所",5713:"",6301:"コマツ",\
            6302:"住友重機械工業",6326:"クボタ",6305:"日立建機",6366:"千代田化工建設",6367:"ダイキン工業",6479:"ミネベア",6770:"アルプス電気",\
            6954:"ファナック",7202:"いすゞ自動車",7270:"富士重工業",8001:"",8002:"丸紅",8015:"豊田通商",8028:"ユニー・ファミリーマートホールディングス",\
            8830:"住友不動産",9007:"小田急電鉄",9009:"京成電鉄",9433:"KDDI",9503:"関西電力",9531:"東京ガス",9983:"ファーストリテイリング",9984:"ソフトバンクグループ" }

names = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/from2000keys.csv")
emp_portfolio_stocks = names[check_main_stock_in_portfolio(sol_emp_output_array,thre=0.05) > 0 ].astype(int)
lasso_portfolio_stocks = names[check_main_stock_in_portfolio(sol_lasso_output_array,thre=0.05) > 0 ].astype(int)
emp_portfolio_stocks_names = np.array([ ticker_dic[name] for name in emp_portfolio_stocks ])
lasso_portfolio_stocks_names = np.array([ ticker_dic[name] for name in lasso_portfolio_stocks ])

for i in np.arange(len(lasso_portfolio_stocks_names)):
    print lasso_portfolio_stocks_names[i]

#Save
#import json
#f = open("/Users/kazeto/Desktop/GradThesis/nikkei/output/alldata.json", "w")
#json.dump(roling_back_up_dict, f)

import pickle
output = open('/Users/kazeto/Desktop/GradThesis/nikkei/output/alldata_window110_rho04_ratio02_norm_thre005_exitnan.pkl', 'wb')
pickle.dump(roling_back_up_dict_thre5, output)
output.close()

#Load
#f = open("/Users/kazeto/Desktop/GradThesis/nikkei/alldata.json")
#backup = json.load(f)

pkl_file = open('/Users/kazeto/Desktop/GradThesis/nikkei/output/alldata_window110_rho04_ratio02.pkl', 'rb')
a = pickle.load(pkl_file)



#######
#1 time of simulation.
traindata, testdata = split_data(data,150)
sol_empirical,r1 = mean_variance_model_optim(traindata,r0=0.01)
sol_emp_output = sol_empirical['x']
sol_lasso,r2 = mean_variance_model_optim(traindata,S=W_new,r0=0.01)
sol_lasso_output = sol_lasso['x']
cvxopt.matrix_repr = printing.matrix_str_default
#cvxopt.spmatrix_repr = printing.spmatrix_str_default
sol_emp = np.array(sol_emp_output)
sol2_lasso = np.array(sol_lasso_output)

test_retrun_emp = np.dot(testdata, sol_emp)
test_return_lasso = np.dot(testdata, sol2_lasso)
print(np.mean(test_retrun_emp))
print(np.mean(test_return_lasso))


###
#TOPIX core 30
"""
core30 = ['2914','3382','4063','4502','4503','6501','6752','6758','6861','6902','6954','6981','7201','7203','7267','7751','8031','8058','8306','8316','8411','8766','8801','8802','9020','9022','9432','9433','9437','9984']

name = np.loadtxt("/Users/kazeto/Desktop/nikkei/names.csv",delimiter=",")
core30_2 = [int(core30[i]) for i in np.arange(len(core30))]

core30_index = [np.where(name == core30_2[i])[0] for i in np.arange(len(core30_2))]
del core30_index[8]
del core30_index[10]
core30_index2 = [core30_index[i][0] for i in np.arange(len(core30_index))]
datap = pd.DataFrame(data)
data_core30 = pd.DataFrame(data).ix[:,core30_index2]
datap.ix[:, datap.columns.isin(core30_index2)]
"""
core = np.loadtxt("/Users/kazeto/Desktop/nikkei/core30_logdiffdata.csv",delimiter=",")
d = np.copy(core)
