import torch
import torch.optim as optim
import math
from torchvision import transforms

from .optimizer import optimize_eq_constrained
from .optimizer import optimize_eq_constrained_adam

class MADGenerator:
    """
    Construct the object.

    @param img: the (distored image)
    @param ref: the reference image
    @param m1: IQA measure 1, for IQAs where the order is important, the
               order is (img, ref)
    @param m2: IQA measure 2
    @param alpha1p: for Adam
    @param alpha1p: for Adam
    @param alpha2p: for Adam
    @param alpha2p: for Adam
    @param lmd: for constrained optimisation

    """
    def __init__(
            self,
            img, ref, m1, m2,
            max_iter=100,
            alpha1p=0.1, alpha1n=-0.1, alpha2p=0.1, alpha2n=-0.1, eps=1e-6, lmd=2):
        self._img = img
        self._ref = ref
        self._m1 = m1
        self._m2 = m2
        self._eps = eps
        self._lmd = lmd
        self._calculated = False
        self._max_iter = max_iter
        self._alpha1p = alpha1p
        self._alpha1n = alpha1n
        self._alpha2p = alpha2p
        self._alpha2n = alpha2n

    def calculate(self):
        img = self._img
        ref = self._ref
        f = lambda x : self._m1(x, ref)
        g = lambda x : self._m2(x, ref)
        with torch.no_grad():
            f0 = f(img)
            g0 = g(img)
        res = {}
        #* res['ps'] = optimize_eq_constrained(self._img, f, g, g0, self._alphas1p, eps=self._eps, lmd=self._lmd)
        #* res['ns'] = optimize_eq_constrained(self._img, f, g, g0, self._alphas1n, eps=self._eps, lmd=self._lmd)
        #* res['sp'] = optimize_eq_constrained(self._img, g, f, f0, self._alphas2p, eps=self._eps, lmd=self._lmd)
        #* res['sn'] = optimize_eq_constrained(self._img, g, f, f0, self._alphas2n, eps=self._eps, lmd=self._lmd)
        res['ps'] = optimize_eq_constrained_adam(self._img, f, g, g0, self._max_iter, self._alpha1p, eps=self._eps, lmd=self._lmd)
        res['ns'] = optimize_eq_constrained_adam(self._img, f, g, g0, self._max_iter, self._alpha1n, eps=self._eps, lmd=self._lmd)
        res['sp'] = optimize_eq_constrained_adam(self._img, g, f, f0, self._max_iter, self._alpha2p, eps=self._eps, lmd=self._lmd)
        res['sn'] = optimize_eq_constrained_adam(self._img, g, f, f0, self._max_iter, self._alpha2n, eps=self._eps, lmd=self._lmd)
        self._res = res

        self._calculated = True

    """
    Get results.

    @param idx: a two character string, 'ps' means positive learning rate
                for m1 and keep m2 the same, etc.
    """
    def __getitem__(self, idx):
        if not self._calculated:
            self.calculate()
        return self._res[idx]


