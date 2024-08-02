from sklearn.base import BaseEstimator, TransformerMixin
from limmaDEGs import *

import pandas as pd
import numpy as np

class PyLimmaDEG(BaseEstimator, TransformerMixin):
    def __init__(self, lfc=1.0, pvalue=0.05, coverage=1, contrasts=None):
        self.lfc = lfc
        self.pvalue = pvalue
        self.coverage = coverage
        self.contrasts = contrasts
        self.selected_genes_ = None

    def fit(self, X, y):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.contrasts = makeContrasts(len(np.unique(y)))
        self.stats, self.selected_genes_ = DEGsExtraction(X.T, y, self.lfc, self.pvalue, self.coverage, self.contrasts)
        return self

    def transform(self, X):
        if self.selected_genes_ is None:
            raise RuntimeError("You must fit the selector before transforming data!")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X.T.iloc[self.selected_genes_].T