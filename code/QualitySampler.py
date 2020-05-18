import numpy as np


class QualitySampler():

    def __init__(self, n_init):
        self.n_init = n_init
        self.good_count = np.zeros(n_init) # assign zero error to init data
        self.runs = 0.0

    def fit_resample(self, X, y):
        self.runs += 1.0
        nl = len(y)
        # Class sampling probabilities
        cp = np.bincount(y.astype(int)) / nl
        minp = np.min(cp)

        cp = minp / cp

        ne = len(self.good_count)
        if nl > ne: # extend good count vector to fit new data
            self.good_count = np.append(self.good_count, np.zeros(nl-ne))

        same_label = np.equal(X[:,-2], X[:,-1])  # check where we have the same label on last two columns (i.e. given and predicted

        selected = []

        for i in np.arange(nl): # we skip counting errors for the init data
            if i < self.n_init:
                self.good_count[i] += 1.0
            elif same_label[i]:
                self.good_count[i] += 1.0

            if np.random.random() < (self.good_count[i]) / self.runs * cp[int(y[i])]:
                selected.append(i)

        return X[selected,:], y[selected]