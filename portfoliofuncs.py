#!/usr/bin/env python
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import networkx as nx
import cvxopt
from cvxopt import solvers,sparse,printing
import sklearn.covariance as cov


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
    X,Y = np.meshgrid(x,y)
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
    #print("ratio of portfolio : {}".format(x))
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


def roling_portfolio(d,r0=0.01, window_size=100, methods='lasso', rho=0.4,lam_rho_ratio=0.2,\
                    inportfolio_thre=0.0, using_sklearn=False):
    """
        input paramater
        ----------------------
        d : ndarray
        r0 : float
            expecting return which the portfolio must satisfy.
        window_size : integer
            the range of window which caluculate ratio of portfolio.
        methods : string 
            methods should be 'empirical' or 'lasso'.
        rho : float
            adding value to diagonal element in coordinate descent algorithm.
        lam_rho_ratio : float
            descide lambda in this regularlization
            (lambda is parameter of strongthness of regularlization)
        inportfolio_thre : float
            If inportfolio_thre is not 0.0 then the stocks under inportfolio_thre
            will be omitted and the ratio of portfolio will be normalized.
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

    cvxopt.matrix_repr = printing.matrix_str_default #for dealing cvxopt matrix as np_matrix.
    if using_sklearn == True:
        model = cov.GraphLasso(alpha=0.01,mode='cd',tol=1e-3)

    for start in np.arange(len(d) - window_size -1):
        print("----------- step : {} -----------".format(start))
        d_window = window_data(d,start,window_size)
        S_window = np.cov(d_window.T,ddof=1)
        N_window = d_window.shape[0]
        M_window = d_window.shape[1]

        if methods == 'lasso' and using_sklearn == True:
            model.fit(d_window)
            W_window = model.covariance_
        elif methods == 'lasso' and using_sklearn == False:
            W_window = cov_lasso_optim(S=S_window,N=N_window,M=M_window,rho=rho,lam_rho_ratio=lam_rho_ratio)

        if methods == 'empirical':
            sol, r = mean_variance_model_optim(d_window,r0=r0)
        elif methods == 'lasso':
            sol, r = mean_variance_model_optim(d_window,S=W_window,r0=r0)
        else:
             raise ValueError("methods should be \'empirical\' or \'lasso\'.")

        sol_output = sol['x']
        testdata = d[start+window_size+1:,:]

        if inportfolio_thre != 0.0:
            sol_norm_output = normalized_ratio(sol_output,thre=inportfolio_thre)
            test_retrun = np.dot(testdata,sol_norm_output)[0]
        else:
            test_retrun = np.dot(testdata,sol_output)[0]

        test_retrun_array.append(test_retrun)

        status_array.append(sol['status'])

        sol_output_vector = np.asarray(sol_output).flatten()
        sol_output_array.append(np.array(sol_output_vector))

        print("N,M : {}, {}".format(N_window,M_window))
        print("Optimal Solution : {}".format(sol['status']))

    #save to dict
    back_up_dict = {}
    back_up_dict['test_return_array'] = np.array(test_retrun_array).flatten()
    back_up_dict['expected_return'] = np.mean(test_retrun_array) * 12
    back_up_dict['risk'] = np.std(test_retrun_array) * 12
    back_up_dict['sol_output_array'] = np.array(sol_output_array)
    back_up_dict['status_array'] = np.array(status_array)
    if methods == 'lasso':
        back_up_dict['rho'] = rho
    if methods == 'lasso' and using_sklearn == True:
        back_up_dict['lam_rho_ratio'] = None
    elif  methods == 'lasso' and using_sklearn == False:
        back_up_dict['lam_rho_ratio'] = lam_rho_ratio
    back_up_dict['window_size'] = window_size
    back_up_dict['r0'] = r0
    back_up_dict['data_dimension'] = d.shape[1]

    if 'unknown' in status_array:
        raise Warning("!!!!!!!!Optimal solution was not found!!!!!!!!!")
    else:
        print("Optimal solution was found in all steps!")

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

def plot_test_return(empirical_test_return, lasso_test_return):
    """
        input paramater
        ----------------------
        empirical_test_return : ndarray
        lasso_test_return : ndarray
        output of roling_portfolio function.

        Ex.
            plot_test_return(emp_roling_dict['test_return_array'], lasso_roling_dict['test_return_array'])
        ---------------------
        returns
        ---------------------
        None
        ---------------------

    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(empirical_test_return, 'b', label="return of Empirical Covariance")
    ax.plot(lasso_test_return, 'r', label="return of Estimated Covariance by LASSO")
    plt.title("Comparison of Retrun")
    ax.legend(loc = 'bottom left')
    plt.ylabel("return of test data")
    plt.xlabel("time")
    fig.show()

if __name__ == '__main__':
    data = np.loadtxt("/Users/kazeto/Desktop/GradThesis/nikkei/logdiffdata.csv",delimiter=",")
    emp_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='empirical',inportfolio_thre=0.01)
    lasso_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='lasso',rho=0.4,\
                                            inportfolio_thre=0.01,using_sklearn=True)
    #my_lasso_roling_dict = roling_portfolio(data,r0=0.01,window_size=110,methods='lasso',rho=0.4,\
    #                                        lam_rho_ratio=0.2,inportfolio_thre=0.01,using_sklearn=False)
    plot_test_return(emp_roling_dict['test_return_array'], lasso_roling_dict['test_return_array'])
