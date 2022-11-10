import pandas as pd
import numpy as np
from typing import List, Tuple
import kqc_custom

from qiskit_optimization import QuadraticProgram
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_random_state
from qiskit_optimization import QuadraticProgram
from qiskit import Aer
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import (
    MinimumEigenOptimizer)

def f_obj(Q,beta) : 
    return -1*np.matmul(np.matmul(beta.T,Q),beta)
def f_nabla(Q,beta) :
    nabla_beta = -2*np.matmul(Q,beta)
    return nabla_beta.values
def l1_subgradient(beta) : return((beta>0)*1 - (beta<0)*1)


class OrdinaryEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2):
        """
        Called when initializing the classifier
        """
        self.r = r

        # THIS IS WRONG! Parameters should have same name as attributes
    
    def fit(self, M, y=None):    
        n = M.shape[0];p=M.shape[1]
        num_iter= 5000
        lr = 0.005
        M_temp = M
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))
        for j in range(self.r):
            theta_temp = np.random.normal(0,1,p)
            for i in range(num_iter) :
                theta_temp += -1*lr*f_nabla(M_temp,theta_temp)
                theta_temp = theta_temp/np.linalg.norm(theta_temp)
            theta_temp = theta_temp.reshape(-1,1)
            lamda = np.matmul(np.matmul(theta_temp.T,M),theta_temp)
            M_temp += -lamda[0][0]*np.matmul(theta_temp,theta_temp.T)
            coef_frame.iloc[:,j] = theta_temp
        self.coef_frame = coef_frame
        return self


    def transform(self, X):
        return self.coef_frame


class SparseEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2, k = 0.5):
        """
        Called when initializing the classifier
        """
        self.r = r
        self.k = k
        # THIS IS WRONG! Parameters should have same name as attributes
    
    def fit(self, M, y=None):    
        n = M.shape[0];p=M.shape[1]
        num_iter= 5000
        lr = 0.005
        M_temp = M
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))
        for j in range(self.r):
            theta_temp = np.random.normal(0,1,p)
            for i in range(num_iter) :
                theta_temp += -1*lr*(f_nabla(M_temp,theta_temp)+ (self.k)*l1_subgradient(theta_temp))
                theta_temp = theta_temp/np.linalg.norm(theta_temp)
            theta_temp = theta_temp.reshape(-1,1)
            lamda = np.matmul(np.matmul(theta_temp.T,M),theta_temp)
            M_temp += -lamda[0][0]*np.matmul(theta_temp,theta_temp.T)
            coef_frame.iloc[:,j] = theta_temp
        self.coef_frame = coef_frame
        return self


    def transform(self, X):
        return self.coef_frame


class BinaryEig(BaseEstimator, TransformerMixin):  
    def __init__(self, r=2, k=-1):
        """
        Called when initializing the classifier
        """
        self.r = r
        self.k = k


    def fit(self, M, y=None):
        """kernel matix, axis, hyperparmeter"""
        Q = -M
        Q_temp = -M
        p = M.shape[0]
        coef_frame = pd.DataFrame(np.zeros((p,self.r)))
        if type(self.k) == int or type(self.k) == float : k_list = [self.k for i in range(self.r)]
        else : k_list = self.k

        for j in range(self.r):
            q = Q_temp.shape[0]
            for i in range(q):
                Q_temp.iloc[i,i] += k_list[j]
            beta = np.zeros((q,1))
            result = kqc_custom.qubo_exact(Q_temp,beta)
            coef_temp = np.array([i for i in result[0]])
            
            rest = 1-coef_frame.apply(sum,1)
            coef_series = pd.DataFrame(np.zeros(p))
            coef_series.loc[[bool(i) for i in rest]] = coef_temp.reshape(-1,1)
            coef_frame.iloc[:,j] = coef_series
            
            rest = 1-coef_frame.apply(sum,1)
            Q_temp = Q.loc[[bool(i) for i in rest],[bool(i) for i in rest]]
        self.coef_frame = coef_frame
        return self

    def transform(self, X):
        return self.coef_frame
    
    
## Solver
def qubo_qaoa(Q,beta,backend = Aer.get_backend("qasm_simulator")):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    quantum_instance = QuantumInstance(backend)
    mes = QAOA(quantum_instance=quantum_instance)
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])

def qubo_exact(Q,beta):
    algorithm_globals.massive = True
    p = Q.shape[0]
    mod = QuadraticProgram("my problem")
    linear = {"x"+str(i): beta[i] for i in range(p)}
    quadratic = {("x"+str(i),"x"+str(j)): Q.values[i,j] for i in range(p) for j in range(p)}

    for i in range(p) :
        mod.binary_var(name="x"+str(i))

    mod.minimize(linear=linear,quadratic=quadratic)
    mes = NumPyMinimumEigensolver()
    optimizer = MinimumEigenOptimizer(mes)
    result = optimizer.solve(mod)
    return([result,mod])


# Sample_Generator

def generate_independent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))
        # normally distributed features
        X = rng.randn(n_samples, n_features)
        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))
        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)
        return X, y

def generate_dependent_sample(n_samples=500, n_features=10, beta_coef =[4,3,2,2],epsilon=4,covariance_parameter=1, random_state=None):

        rng = check_random_state(random_state)
        if n_features < 4:
            raise ValueError("`n_features` must be >= 4. "
                             "Got n_features={0}".format(n_features))

        # normally distributed features

        v = rng.normal(0, 0.4, (n_features, n_features))
        mean = np.zeros(n_features)
        cov = v @ v.T*covariance_parameter + 0.1 * np.identity(n_features)
        X = rng.multivariate_normal(mean, cov, n_samples)

        # beta is a linear combination of informative features
        n_informative = len(beta_coef)
        beta = np.hstack((
            beta_coef, np.zeros(n_features - n_informative)))

        # cubic in subspace
        y = np.dot(X, beta)
        y += epsilon * rng.randn(n_samples)

        return X, y


def read_otu(task_dir, otu_dir, positive_value):
    task = pd.read_csv(task_dir, '\t')
    otutable = pd.read_csv(otu_dir, '\t')
    samples = list(task['#SampleID'])
    data = dict()
    for sample in samples:
        otu = list(otutable[sample])
        otu.append(task.set_index('#SampleID').transpose()[sample][0])
        data[sample]=otu
    df = pd.DataFrame(data).transpose()
    df.columns = df.columns.map(lambda x: 'OTU_'+str(x+1))
    y = df["OTU_" + str(df.shape[1])].map(lambda x : 1 if x==positive_value else 0)
    X = df.drop("OTU_" + str(df.shape[1]), axis=1)
    return(X, y)
