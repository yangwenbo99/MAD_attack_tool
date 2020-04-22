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
    @param alphas1p: an iterative for generating learning rate when
                     increasing m1
    @param alphas1p: an iterative for generating learning rate when
                     decreasing m1
    @param alphas2p: an iterative for generating learning rate when
                     increasing m2
    @param alphas2p: an iterative for generating learning rate when
                     decreasing m2
    """
    def __init__(
            self,
            img, ref, m1, m2,
            alphas1p, alphas1n, alphas2p, alphas2n, eps=1e-6, lmd=20):
        self._img = img
        self._ref = ref
        self._m1 = m1
        self._m2 = m2
        self._alphas1p = alphas1p
        self._alphas1n = alphas1n
        self._alphas2p = alphas2p
        self._alphas2n = alphas2n
        self._eps = eps
        self._lmd = lmd
        self._calculated = False

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
        res['ps'] = optimize_eq_constrained_adam(self._img, f, g, g0, 100, 0.3, eps=self._eps, lmd=self._lmd)
        res['ns'] = optimize_eq_constrained_adam(self._img, f, g, g0, 100, -0.3, eps=self._eps, lmd=self._lmd)
        res['sp'] = optimize_eq_constrained_adam(self._img, g, f, f0, 100, 0.3, eps=self._eps, lmd=self._lmd)
        res['sn'] = optimize_eq_constrained_adam(self._img, g, f, f0, 100, -0.3, eps=self._eps, lmd=self._lmd)
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


