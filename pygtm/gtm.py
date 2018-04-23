import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA


class GTM(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_rbfs=10, sigma=1, alpha=1e-3, n_grids=20, method='mean',
                 max_iter=10, tol=1e-3, random_state=None, verbose=False):
        self.n_components = n_components
        self.n_rbfs = n_rbfs
        self.sigma = sigma
        self.alpha = alpha
        self.n_grids = n_grids
        self.max_iter = max_iter
        self.method = method
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.prev_likelihood_ = -float('inf')
    
    def get_lattice_points(self, n_grid):
        grid = np.meshgrid(*[np.linspace(-1, 1, n_grid + 1) for _ in range(self.n_components)])
        return np.array([c.ravel() for c in grid]).T
    
    def init(self, X):
        # generate map
        self.z = self.get_lattice_points(self.n_grids)
        self.rbfs = self.get_lattice_points(self.n_rbfs)
        d = cdist(self.z, self.rbfs, 'sqeuclidean')
        self.phi = np.exp(-d / (2 * self.sigma))
        
        # init W and beta from PCA
        pca = PCA(n_components=self.n_components + 1, random_state=self.random_state)
        pca.fit(X)
        self.W = np.linalg.pinv(self.phi).dot(self.z).dot(pca.components_[:self.n_components, :])
        
        betainv1 = pca.explained_variance_[self.n_components]
        inter_dist = cdist(self.phi.dot(self.W), self.phi.dot(self.W))
        np.fill_diagonal(inter_dist, np.inf)
        betainv2 = inter_dist.min(axis=0).mean() / 2
        self.beta = 1 / max(betainv1, betainv2)
    
    def responsibility(self, X):
        p = np.exp((-self.beta / 2) * cdist(self.phi.dot(self.W), X, 'sqeuclidean'))
        return p / p.sum(axis=0)
    
    def likelihood(self, X):
        R = self.responsibility(X)
        D = X.shape[1]
        k1 = (D / 2) * np.log(self.beta / (2 * np.pi))
        k2 = -(self.beta / 2) * cdist(self.phi.dot(self.W), X, 'sqeuclidean')
        return (R * (k1 + k2)).sum()
    
    def fit(self, X, y=None, **fit_params):
        self.init(X)
        
        for i in range(self.max_iter):
            R = self.responsibility(X)
            G = np.diag(R.sum(axis=1))
            self.W = np.linalg.solve(
                self.phi.T.dot(G).dot(self.phi) + (self.alpha / self.beta) * np.identity(self.phi.shape[1]),
                self.phi.T.dot(R).dot(X))
            
            self.beta = X.size / (cdist(self.phi.dot(self.W), X, 'sqeuclidean') * R).sum()
            
            likelihood = self.likelihood(X)
            diff = abs(likelihood - self.prev_likelihood_) / X.shape[0]
            self.prev_likelihood_ = likelihood
            if self.verbose:
                print('cycle #{}: likelihood: {:.3f}, diff: {:.3f}'.format(i + 1, likelihood, diff))
            
            if diff < self.tol:
                if self.verbose:
                    print('converged.')
                break
    
    def transform(self, X, y=None):
        assert self.method in ('mean', 'mode')
        if self.method == 'mean':
            R = self.responsibility(X)
            return self.z.T.dot(R).T
        elif self.method == 'mode':
            return self.z[self.responsibility(X).argmax(axis=0), :]
    
    def inverse_transform(self, Xt):
        d = cdist(Xt, self.rbfs, 'sqeuclidean')
        phi = np.exp(-d / (2 * self.sigma))
        return phi.dot(self.W)
