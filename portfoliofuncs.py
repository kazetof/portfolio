#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import cvxopt
from cvxopt import solvers,sparse,printing
import sklearn.covariance as cov
import pickle

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
    This function is for dubugging in the estimation steps of
    covariance matrix. If there is inf in covariance matrix,
    this function will alert it and the index where inf exists.
    
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
        beta_new[k] = soft_thre(term1,lam) * term2
        beta_old = np.copy(beta_new)

    W_12_new = np.dot(W_11,beta_new)
    W_newcolum = np.insert(W_12_new,[num],np.diag(W)[num])
    W_new = np.copy(W)
    W_new[:,num] = W_newcolum
    W_new[num,:] = W_newcolum
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

            W_new = update_W12(W_old,S,i,rho,lam=rho*lam_rho_ratio)
            W_old = np.copy(W_new)
    return W_new

def get_edge_matrix(W):
    """
    This function return just 0 or 1 matrix to plot 
    network graph.

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
    This function will return correration matrix.
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
    X,Y = np.meshgrid(x[::-1],y)
    fig, ax = plt.subplots()
    ax.pcolor(X,Y,matrix)
    plt.title(title)
    #ax.pcolor(X,Y,matrix, cmap=plt.cm.Blues)
    fig.show()

def extract_nonzero(matrix):
    """
    This function return the submatrix which the non diagonal elements are not zero.
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
    #print("propotion of portfolio : {}".format(x))
    print("portfolio return is {}".format(np.sum(cvxopt.mul(x,cvxopt.matrix(r)))))
    print("sum of propotion x is {}".format(np.sum(x)))
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


def roling_portfolio(d,r0=0.01, window_size=100, methods='lasso', rho=0.4,lam_rho_ratio=0.2,\
                    inportfolio_thre=0.0, using_sklearn_glasso=False, shrunk_param=None):
    """
        input paramater
        ----------------------
        d : ndarray
        r0 : float
            expecting return which the portfolio must satisfy.
        window_size : integer
            the range of window which caluculate propotion of portfolio.
        methods : string 
            methods should be 'empirical', 'lasso', 'shrunk',
             'empirical_diag' or 'singleindex'.
        rho : float
            It must be in [0,1]
            adding value to diagonal element in coordinate descent algorithm.
        lam_rho_ratio : float
            It must be in [0,1]
            descide lambda in this regularlization
            (lambda is parameter of strongthness of regularlization)
        inportfolio_thre : float
            If inportfolio_thre is not 0.0 then the stocks under inportfolio_thre
            will be omitted and the propotion of portfolio will be normalized.
        using_sklearn_glasso : bool
        shrunk_param : float
            The shrinkage parameter in shrunk method.
            It must be in [0,1]
        ---------------------

        returns
        ---------------------
        back_up_dict : dictionary
            test_retrun_emp_array ; return in test data for portfolio which used empirical covariance matrix.
            test_return_lasso_array ; return in test data for portfolio which used lasso covariance matrix.
        ---------------------

    """
    #define list to save
    test_retrun_array = []
    sol_output_array = []
    status_array = []
    #d_window_mean = []
    #d_window_variance = []

    cvxopt.matrix_repr = printing.matrix_str_default #for dealing cvxopt matrix as np_matrix.

    if methods == 'lasso' and using_sklearn_glasso == True:
        if shrunk_param == None:
            shrunk_param = 0.01
            model = cov.GraphLasso(alpha=shrunk_param, mode='cd', tol=1e-3, assume_centered=False)
        else:
            model = cov.GraphLasso(alpha=shrunk_param, mode='cd', tol=1e-3, assume_centered=False)
    elif methods == 'shrunk':
        if shrunk_param == None:
            shrunk_param = 0.6
            model = cov.ShrunkCovariance(shrinkage=shrunk_param, assume_centered=False)
        else:
            model = cov.ShrunkCovariance(shrinkage=shrunk_param, assume_centered=False)

    for start in np.arange(len(d) - window_size -1):
        print("----------- step : {} -----------".format(start))
        d_window = window_data(d,start,window_size)
        N_window = d_window.shape[0]
        M_window = d_window.shape[1]

        if methods == 'empirical':
            W_window = np.cov(d_window.T,ddof=1)
        elif methods == 'lasso' and using_sklearn_glasso == True:
            model.fit(d_window)
            W_window = model.covariance_
        elif methods == 'lasso' and using_sklearn_glasso == False:
            S_window = np.cov(d_window.T,ddof=1)
            W_window = cov_lasso_optim(S=S_window,N=N_window,M=M_window,rho=rho,lam_rho_ratio=lam_rho_ratio)
        elif methods == 'shrunk':
            model.fit(d_window)
            W_window = np.linalg.inv(model.get_precision())
        elif methods == 'empirical_diag':
            W_window = np.diag(np.diag(np.cov(d_window.T)))
        elif methods == 'singleindex':
            W_window = make_single_index_covariance_matrix(d_window)
        else:
            raise ValueError("methods should be \'empirical\', \'lasso\', \'shrunk\', \'empirical_diag\' or \'singleindex\'.")

        sol, r = mean_variance_model_optim(d_window, S=W_window, r0=r0)
        sol_output = sol['x']
        testdata = d[start+window_size+1:,:]

        if inportfolio_thre != 0.0:
            sol_norm_output = normalized_propotion(sol_output,thre=inportfolio_thre)
            test_retrun = np.dot(testdata,sol_norm_output)[0]
        else:
            test_retrun = np.dot(testdata,sol_output)[0]

        test_retrun_array.append(test_retrun)
        status_array.append(sol['status'])

        sol_output_vector = np.asarray(sol_output).flatten()
        sol_output_array.append(np.array(sol_output_vector))

        #d_window_mean.append(np.mean(d_window))
        #d_window_variance.append(np.var(d_window))

        print("N,M : {}, {}".format(N_window,M_window))
        print("Optimal Solution : {}".format(sol['status']))

    #save to dict
    back_up_dict = {}
    back_up_dict['test_return_array'] = np.array(test_retrun_array).flatten()
    back_up_dict['expected_return'] = np.mean(test_retrun_array)
    back_up_dict['risk'] = np.std(test_retrun_array)
    back_up_dict['sol_output_array'] = np.array(sol_output_array)
    if methods == 'lasso':
        back_up_dict['rho'] = rho
    if methods == 'lasso' and using_sklearn_glasso == True:
        back_up_dict['lam_rho_ratio'] = None
    elif  methods == 'lasso' and using_sklearn_glasso == False:
        back_up_dict['lam_rho_ratio'] = lam_rho_ratio
    back_up_dict['window_size'] = window_size
    back_up_dict['r0'] = r0
    back_up_dict['data_dimension'] = d.shape[1]
    back_up_dict['methods'] = methods
    #back_up_dict['d_window_mean'] = np.array(d_window_mean)
    #back_up_dict['d_window_variance'] = np.array(d_window_variance)
    if shrunk_param != None:
        back_up_dict['shrunk_param'] = shrunk_param

    #back_up_dict['optimal_status'] = np.array(status_array)
    if 'unknown' in status_array:
        raise Warning("!!!!!!!!Optimal solution was not found!!!!!!!!!")
        back_up_dict['optimal_status'] = 'not optimal'
    else:
        print("Optimal solution was found in all steps!")
        back_up_dict['optimal_status'] = 'optimal'

    return back_up_dict

def equal_rolling_portfolio(data, window_size=110):
    """
        input paramater
        ----------------------
        data : ndarray
            (n*p) matrix
        window_size : int
           the range of window which caluculate propotion of portfolio.
        ---------------------

        returns
        ---------------------
        back_up_dict : dictionary
            the same as rolling_portfolio.
        ---------------------

    """
    back_up_dict = {}

    return_vec = make_market_portfolio_return(data)
    test_return = return_vec[window_size:data.shape[0]-1].flatten()
    back_up_dict['window_size'] = window_size
    back_up_dict['test_return_array'] = test_return
    back_up_dict['expected_return'] = np.mean(test_return)
    back_up_dict['risk'] = np.std(test_return)
    back_up_dict['data_dimension'] = data.shape[1]
    back_up_dict['methods'] = 'equal'

    return back_up_dict


def check_turnover(output_array,thre=0.01,num_return=True):
    """
        input paramater
        ----------------------
        output_array : ndarray
        propotion of portfolio which is output of quadratic programming.
        thre : float
        If a propotion of portfolio is over thre, the stock will be bought.
        If a propotion of portfolio is under thre, the stock will be sold.
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

def normalized_propotion(propotion_vector, thre=0.01):
    """
        input paramater
        ----------------------
        propotion_vector : ndarray
            n*1 vector
            propotion of portfolio which is output of quadratic programming.
        thre : float
            The thereshuld of portfolio.
            The stocks which is under thre will be 0 and another stocks
            will be normalized to be sum is ten.
        ---------------------

        returns
        ---------------------
        propotion_vector_normalized : ndarray
        n*1 vector
        ---------------------

    """
    if isinstance(propotion_vector, cvxopt.base.matrix) == True:
        propotion_vector = np.array(propotion_vector).flatten()
    input_vector = propotion_vector.copy() #input_vector[bool_vector] = 0.0 will change original vector so copy it.
    bool_vector = input_vector < thre
    input_vector[bool_vector] = 0.0

    if np.sum(input_vector) == 0:
        propotion_vector_normalized = input_vector #avoiding nan
    else:
        propotion_vector_normalized = input_vector / np.sum(input_vector)
    return propotion_vector_normalized

def check_abs_change_portfolio(output_array,vector_return_bool=False):
    """
        input paramater
        ----------------------
        output_array : ndarray
        n*p matrix (n is time and p is number of stocks)
        propotion of portfolio which is output of quadratic programming.

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
    A = np.r_[np.array([np.zeros(output_array.shape[1])]), output_array] #add zero on the top
    B = np.r_[output_array, np.array([np.zeros(output_array.shape[1])])] #add zero at bottom
    C = B - A
    D = C[1:C.shape[0]-1,:] #D is (n-1 * p matrix) since taking difference of t and t-1
    abs_change_vector = np.sum(np.abs(D),0)
    abs_change = np.sum(abs_change_vector)
    if vector_return_bool == True:
        return abs_change_vector
    else:
        return abs_change

def plot_test_return(*dicts):
    """
        input paramater
        ----------------------
        dicts : variable length of dicts which is output of roling_portfolio function.
        Here we assume arguments of methods is different except for 'lasso'.
        
        Ex. plot_test_return(emp_roling_dict, lasso_roling_dict, shrunk_roling_dict)
        ---------------------
        returns
        ---------------------
        None
        ---------------------
    """
    argnum = len(dicts)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for arg_i in np.arange(argnum):
        if dicts[arg_i]['methods'] != 'lasso':
            ax.plot(dicts[arg_i]['test_return_array'], 'b', label="return of {} Covariance".format(dicts[arg_i]['methods']), color=plt.cm.jet(arg_i*(1./argnum)))
        else:
            label = dicts[arg_i]['methods'] + "( lambda : " + str(dicts[arg_i]['shrunk_param']) + " )"
            ax.plot(dicts[arg_i]['test_return_array'], 'b', label="return of {} Covariance".format(label), color=plt.cm.jet(arg_i*(1./argnum)))
    plt.title("Comparison of Retrun")
    ax.legend(loc = 'upper center',bbox_to_anchor=(0.5,-0.25))
    plt.subplots_adjust(bottom=0.4)
    plt.ylabel("return of test data")
    plt.xlabel("time")
    fig.show()

def plot_turnover(*dicts):
    """
        input paramater
        ----------------------
        dicts : variable length of dicts which is output of roling_portfolio function.
        Here we assume arguments of methods is different except for 'lasso'.

        Ex. turnover_plot(emp_roling_dict, lasso_roling_dict, shrunk_roling_dict)
        ---------------------
        returns
        ---------------------
        None
        ---------------------
    """
    argnum = len(dicts)
    range_list = np.arange(0.01,0.1,0.005)
    turnover_dic = {}
    label_list = []
    for arg_i in np.arange(argnum):
        if dicts[arg_i]['methods'] != 'lasso':
            label = dicts[arg_i]['methods']
            label_list.append(label)
            turnover_dic[dicts[arg_i]['methods']] = np.array([ check_turnover(dicts[arg_i]['sol_output_array'],thre=i,num_return=True) for i in range_list ])
        else:
            label = dicts[arg_i]['methods'] + str(dicts[arg_i]['shrunk_param'])
            label_list.append(label) #.kyes() is OK but in order to align in order, we have to make label_list.
            turnover_dic[label] = np.array([ check_turnover(dicts[arg_i]['sol_output_array'],thre=i,num_return=True) for i in range_list ])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for arg_i in np.arange(argnum):
        ax.plot(range_list,turnover_dic[label_list[arg_i]], label=label_list[arg_i], color=plt.cm.jet(arg_i*(1./argnum)))
    ax.legend()
    plt.xlabel("threshould of portfolio")
    plt.ylabel("number of turnover")
    plt.title("number of turnover")
    fig.show()


def plot_abs_change(*dicts):
    """
        input paramater
        ----------------------
        dicts : variable length of dicts which is output of roling_portfolio function.
        Here we assume arguments of methods is different except for 'lasso'.

        Ex. plot_abs_change(emp_roling_dict, lasso_roling_dict, shrunk_roling_dict)
        ---------------------
        returns
        ---------------------
        None
        ---------------------
    """
    argnum = len(dicts)
    thre_range = np.arange(0.01,0.05,0.0025)
    
        #make label list for legend
    label_list = []
    for arg_i in np.arange(argnum):
        if dicts[arg_i]['methods'] != 'lasso':
            label = dicts[arg_i]['methods']
            label_list.append(label)
        else:
            label = dicts[arg_i]['methods'] + str(dicts[arg_i]['shrunk_param'])
            label_list.append(label)

    abs_change_dic = {}
    for arg_i in np.arange(argnum):
        abs_change_dic[label_list[arg_i]] = np.array([ check_abs_change_portfolio(normalized_propotion_array(dicts[arg_i]['sol_output_array'],thre=i)) for i in thre_range ])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    for arg_i in np.arange(argnum):
        ax1.plot(thre_range, abs_change_dic[label_list[arg_i]], label=label_list[arg_i], color=plt.cm.jet(arg_i*(1./argnum)))
    plt.title("Abs values of change")
    plt.xlabel("propotion of threshold")
    plt.ylabel("sum of absolute value of change")
    plt.legend(loc="upper left")
    fig.show()


def plot_stock_num(thre=0.01,*dicts):
    """
        input paramater
        ----------------------
        thre : float
            threshold of portfolio propotion.
        dicts : variable length of dicts which is output of roling_portfolio function.
            Here we assume arguments of methods is different except for 'lasso'.

        Ex. plot_stock_num(0.01, emp_roling_dict, lasso_roling_dict, shrunk_roling_dict)
        ---------------------
        returns
        ---------------------
        None
        ---------------------
    """
    argnum = len(dicts)

    #make label list for legend
    label_list = []
    for arg_i in np.arange(argnum):
        if dicts[arg_i]['methods'] != 'lasso':
            label = dicts[arg_i]['methods']
            label_list.append(label)
        else:
            label = dicts[arg_i]['methods'] + str(dicts[arg_i]['shrunk_param'])
            label_list.append(label)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for arg_i in np.arange(argnum):
        num_vector = check_stock_num_in_portfolio(dicts[arg_i],thre=thre)
        ax.plot(np.arange(len(num_vector)), num_vector, label=label_list[arg_i], color=plt.cm.jet(arg_i*(1./argnum)))
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    plt.xlabel("time")
    plt.subplots_adjust(right=0.6)
    plt.ylabel("number of stocks")
    plt.title("number of stocks in portfolio")
    fig.show()


def check_main_stock_in_portfolio(output_dict,thre=0.01):
    """
    This function returns how many times the stock will in portfolio for each stock.
    """
    bool_array = output_dict['sol_output_array'] > thre
    return np.sum(bool_array,0)

def check_stock_num_in_portfolio(output_dict,thre=0.01):
    """
    This function returns how many stocks the portfolio contains for each time.
    """
    bool_array = output_dict['sol_output_array'] > thre
    return np.sum(bool_array,1)

def normalized_propotion_array(output_array,thre=0.01):
    output_norm = np.array([ normalized_propotion(output_array[i],thre=thre) for i in np.arange(output_array.shape[0]) ])
    return output_norm

def save_dic(dic, PATH):
    """
        input paramater
        ----------------------
        dic : dictionary
            output of roling_portfolio function.
        PATH : string
        ---------------------
        returns
        ---------------------
        None
        ---------------------

    """
    output = open(PATH, 'wb')
    pickle.dump(dic, output)
    output.close()

def load_dic(PATH):
    """
        input paramater
        ----------------------
        PATH : string
        ---------------------
        returns
        ---------------------
        dic : dictionary
        ---------------------

    """
    pkl_file = open(PATH, 'rb')
    dic = pickle.load(pkl_file)
    return dic

def evaluation(output_dict):
    print("mthod is {}".format(output_dict['methods']))
    print("Expected Test Return : {}".format(output_dict['expected_return']))
    print("Risk : {}".format(output_dict['risk']))


#single index model
from sklearn import linear_model
def make_market_portfolio_return(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        market_return : ndarray
            (n * 1) vector
            This is return of market portfolio which has equal propotion.
        ---------------------
    """
    try:
    #If data.shape = (n,), then change it to (n, 1).
        data.shape[1]
    except IndexError: 
        data = np.copy(np.array([data]).T)

    if data.shape[1] == 1:
        equal_weight = np.array([np.ones(data.shape[0]) / data.shape[0]]).T
        market_return = np.array([np.dot(data.T,equal_weight)]).T
        return market_return[0][0][0]
    else:
        equal_weight = np.ones(data.shape[1]) / data.shape[1]
        market_return = np.array([np.dot(data,equal_weight)]).T
        return market_return

def make_single_index_diagonal_covariance_matrix(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        S_single_index_diag : ndarray
            (p * p) matrix
            This is diagonal covariance matrix estimated by single index model
            so this is diagonal matrix.
        ---------------------
    """
    S_single_index = make_single_index_covariance_matrix(data)
    S_single_index_diag = np.diag(np.diag(S_single_index))
    return S_single_index_diag

def make_single_index_covariance_matrix(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        S_single_index : ndarray
            (p * p) matrix
            This is covariance matrix estimated by single index model.
        ---------------------
    """
    def make_single_index_beta_and_varresid(indiv_return,market_return):
        try:
            #If indiv_return.shape = (199,), then change it to (199, 1).
            indiv_return.shape[1]
        except IndexError: 
            indiv_return = np.array([indiv_return]).T

        model = linear_model.LinearRegression()
        model.fit(market_return, indiv_return)
        beta = model.coef_[0][0]
        resid = indiv_return - model.predict(indiv_return)
        var_resid = np.var(resid)

        return beta,var_resid

    market_return = make_market_portfolio_return(data)
    var_market = np.var(market_return)
    beta_vector = np.array([[ make_single_index_beta_and_varresid(data[:,i],market_return)[0] for i in np.arange(data.shape[1])]]).T
    varresid_vector = np.array([ make_single_index_beta_and_varresid(data[:,i],market_return)[1] for i in np.arange(data.shape[1])])
    S_single_index = np.dot(beta_vector,beta_vector.T) * var_market + np.diag(varresid_vector)
    return S_single_index

def plot_cov_glasso_each_lambda(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        None 
            This will plot heat map of covariance matrix estimated by glasso in some graphs.
        ---------------------

    """
    for i in np.arange(0.0005,0.01,0.001):
        model = cov.GraphLasso(alpha=i, mode='cd', tol=1e-3, assume_centered=False)
        model.fit(data)
        S = model.covariance_
        heatmap(S, title="Heat map of Covariance matrix : shrink param = {}".format(str(i)))

def shrunk_param_optim(data):
    r0 = 0.01
    window_size = 110
    inportfolio_thre = 0.01
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

import time
def lasso_param_optim(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        lasso_optim_dict : dict
            This is the dictionary which has dictionaries of 
            roling_portfolio function output at some values of lambda.
        lasso_param_range : ndarray
            The range array of regularization parameter lamnbda.
        ---------------------
    """
    #measure the caluculation time 
    start_time = time.time()

    #define params
    r0 = 0.01
    window_size = 110
    inportfolio_thre = 0.01
    lasso_param_range = np.arange(0.0005,0.01,0.0002)
    print("length of range is {}".format(len(lasso_param_range)))
    lasso_optim_dict = {}
    step = 1

    for i in lasso_param_range:
        print("------ lambda : {} ------".format(i))
        each_start_time = time.time()
        lasso_optim_dict[str(i)] = roling_portfolio(data,r0=r0,window_size=window_size,\
                                        methods='lasso',using_sklearn_glasso=True,\
                                        inportfolio_thre=inportfolio_thre, shrunk_param=i)

        #print each step time
        each_elapsed_time = time.time() - each_start_time
        print("{} step of elapsed time : {}".format(step, each_elapsed_time)) + " sec"
        step += 1

    all_elapsed_time = time.time() - start_time
    print("all of elapsed time : {}".format(all_elapsed_time)) + " sec"

    for i in lasso_param_range:
        print(i)
        print(lasso_optim_dict[str(i)]['expected_return'])
        print(lasso_optim_dict[str(i)]['risk'])

    return lasso_optim_dict, lasso_param_range

def plot_lasso_param_optim_return(lasso_optim_dict, lasso_param_range):
    """
        input paramater
        ----------------------
        lasso_optim_dict :  dictionary
            This is the dictionary which has roling_portfolio function
            output at some values of lambda.
        lasso_param_range : ndarray
            The range array of regularization parameter lamnbda.
        ---------------------
        returns
        ---------------------
        None
            This will plot the test expectation return 
            at some values lambda.
        ---------------------
    """
    r = np.array([ lasso_optim_dict[str(i)]['expected_return'] for i in lasso_param_range ])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lasso_param_range,r)
    plt.xlabel('lambda')
    plt.ylabel('Expected Return')
    plt.title('Return')

def plot_lasso_param_optim_risk(lasso_optim_dict, lasso_param_range):
    """
        input paramater
        ----------------------
        lasso_optim_dict :  dictionary
            This is the dictionary which has roling_portfolio function
            output at some values of lambda.
        lasso_param_range : ndarray
            The range array of regularization parameter lamnbda.
        ---------------------
        returns
        ---------------------
        None
            This will plot the test risk at some values lambda.
        ---------------------
    """
    risk = np.array([ lasso_optim_dict[str(i)]['risk'] for i in lasso_param_range ])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(lasso_param_range,risk)
    plt.xlabel('lambda')
    plt.ylabel('Risk')
    plt.title('Risk (variance of test return)')

def min_risk_lambda(lasso_optim_dict, lasso_param_range):
    """
        input paramater
        ----------------------
        lasso_optim_dict :  dictionary
            This is the dictionary which has roling_portfolio function
            output at some values of lambda.
        lasso_param_range : ndarray
            The range array of regularization parameter lamnbda.
        ---------------------
        returns
        ---------------------
        min_lambda : float
            The value of lambda where the minimun risk.
        ---------------------
    """
    risk = np.array([ lasso_optim_dict[str(i)]['risk'] for i in lasso_param_range ])
    np.where(risk == np.min(risk))
    min_lambda = lasso_param_range[np.where(risk == np.min(risk))[0][0]]
    return min_lambda

def plot_cov_glasso_each_lambda_in_one_graph(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n * p) matrix
        ---------------------
        returns
        ---------------------
        None
            This will plot heat map of covariance matrix estimated by glasso in one graph.
        ---------------------
    """
    ax_list = [221,222,223,224]
    ax_i = 0
    fig = plt.figure()

    for i in np.arange(0.0,0.01,0.0025):
        title= "lambda : " + str(i)
        model = cov.GraphLasso(alpha=i, mode='cd', tol=1e-3, assume_centered=False)
        model.fit(data)
        S = model.covariance_
        x = np.arange(S.shape[0])
        y = np.arange(S.shape[1])
        X,Y = np.meshgrid(x[::-1],y)
        ax = fig.add_subplot(ax_list[ax_i])
        ax.pcolor(X,Y,S)
        plt.title(title)
        ax_i += 1
    fig.show()

def plot_efficient_frontier(data):
    """
        input paramater
        ----------------------
        data : ndarray
            (n*p) matrix
        ---------------------
        returns
        ---------------------
        None
            plot the efficient flonier plot.
        ---------------------
    """
    m = data.mean(0)
    std = data.std(0)
    covmat = np.cov(data, rowvar=0, bias=0)
    flontier = pd.DataFrame(efficient_frontier(data, m, covmat),columns=["mean","std"])
    df = pd.DataFrame(data=np.c_[m,std],index=np.arange(len(m)),columns=["mean","std"])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x=df["std"],y=df["mean"])
    plt.xlabel("std")
    plt.ylabel("mean")
    plt.title("Efficient Frontier")
    ax.plot(flontier['std'],flontier['mean'],c="r")
    fig.show()

def efficient_frontier(data, m, covmat):
    """
        input paramater
        ----------------------
        data : ndarray
            (n*p) matrix
        m : mean vector 
            (p*1) matrix
        covmat : covariance matrix
            (p*p) matrix
        ---------------------
        returns
        ---------------------
        flontier_array : ndarray
            (100*2) array
            100 is the number of r0 range list.
            The columuns has mean and std.
        ---------------------
    """
    def return_portfolio_mean(x,m):
        #x is weight of portfolio.
        weight = np.array(x).flatten()
        p_mean = np.dot(weight.T,m)
        return p_mean

    def return_portfolio_std(x,covmat):
        weight = np.array(x).flatten()
        p_std = np.sqrt(np.dot(np.dot(weight.T,covmat),weight))
        return p_std

    def return_weight(data,r0):
        sol,x = mean_variance_model_optim(data,r=None,S=None,r0=r0)
        return x

    r0_list = np.linspace(0.005,0.01032242,num=100)
    portfolio_mean = np.array([ return_portfolio_mean(return_weight(data,r0=i),m) for i in r0_list ])
    portfolio_std = np.array([ return_portfolio_std(return_weight(data,r0=i),covmat) for i in r0_list ])
    flontier_array = np.c_[portfolio_mean, portfolio_std]
    return flontier_array


if __name__ == '__main__':
    data = np.loadtxt("./logdiffdata.csv",delimiter=",")
    emp_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='empirical',inportfolio_thre=0.01)
    lasso_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='lasso',rho=0.4,\
                                            inportfolio_thre=0.01,using_sklearn_glasso=True)
    #my_lasso_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='lasso',rho=0.4,\
    #                                        lam_rho_ratio=0.2,inportfolio_thre=0.01,using_sklearn_glasso=False)
    plot_test_return(emp_roling_dict['test_return_array'], lasso_roling_dict['test_return_array'])

