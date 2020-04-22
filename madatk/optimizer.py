import torch
import torch.optim as optim
import math
from torchvision import transforms

DEBUG = 1

def get_pritnable(t):
    if t is torch.Tensor:
        return t.clone().detach().cpu().numpy()
    else:
        return t

def img_extra_constrain(x):
    x = x.clone().detach()
    x[x > 1] = 1
    x[x < 0] = 0
    return x

"""
Correct equality constrain

@param x: current x
@param consf: constrain function
@param consdv: destination value of the constrain funciton
@param lmd: step size
@tol: tolerance
"""
class EqualConstrainCorrecter:
    def __init__(self, consf, consdv, initlmd=1, tol=3e-4):
        self._consf = consf
        self._consdv = consdv
        self._lmd = initlmd
        self._tol = tol

    def __call__(self, x):
        def bound(a, bound):
            if a > bound:
                a = bound
            if a < - bound:
                a = -bound
            return a

        x = x.clone().detach()
        x.requires_grad_(True)
        if x.grad is not None:
            x.grad.data.zero_()
        tmp = self._consf(x)
        diff = tmp - self._consdv
        if diff * self._lmd < 0:
            self._lmd *= -1
        while abs(diff) > self._tol:
            if DEBUG > 3:
                print(
                        '\n+',
                        get_pritnable(self._lmd),
                        get_pritnable(diff),
                        # torch.dist(x, x1)
                        )

            tmp.backward()
            fp = x.grad.clone().detach()
            diff = diff.detach()
            with torch.no_grad():
                #* bd = bound(diff, 5 * self._tol)
                #* ss = bd * self._lmd         # step size
                ss = self._lmd
                x1 = x - ss * fp
                ndiff = self._consf(x1) - self._consdv       # new difference
                if abs(ndiff - diff) < 1e-15:
                    if DEBUG:
                        print("\nWarning: no difference", torch.dist(x, x1), ndiff, diff)
                        print(ss)
                    break
                newss = diff / (diff - ndiff) * ss
                x = x - newss * fp
                #* self._lmd = max(min(newss, 20), 0.001)   # numerical issues
                self._lmd = newss
                x = img_extra_constrain(x)
            x.requires_grad_(True)
            if x.grad is not None:
                x.grad.data.zero_()
            tmp = self._consf(x)
            diff = tmp - self._consdv

        return x


"""
Class for supplying learning rates during traininig

Usage:
    x = SequntialAlphaer( (
        (10, 1),
        (100, 0.1)
    ))
    for alpha in x:
        # Then, x will supplies 1 for 10 times, then 0.1 for 100 times
        pass

"""
class SequntialAlphaer:
    def __init__(self, values):
        self._values = values

    def __iter__(self):
        for item in self._values:
            count = 0
            while count < item[0]:
                yield item[1]
                count += 1

    def __getitem__(self, i):
        last = None
        bound = 0
        for item in self._values:
            bound += item[0]
            if i < bound:
                last = item[1]
            else:
                break
        return last


"""
Gradiant accend f with the constrain g(x) = gdv
"""
def optimize_eq_constrained(
        x, f, g, gdv,
        alphas=SequntialAlphaer((200, 0.001)),
        tol=3e-4, lmd=1,
        eps=1e-5):
    ori_img = x.clone()
    x = x.clone()
    correcter = EqualConstrainCorrecter(g, gdv, lmd, tol)

    lastfv = 0
    i = 0
    for lr in alphas:
        x.requires_grad_(True)
        print('\r{:10} {:3} '.format(i, lr),  end='')

        if x.grad is not None:
            x.grad.data.zero_()
        fv = f(x)
        print('', fv.clone().detach().cpu().numpy(), end='')
        if DEBUG >= 4:
            print()
        fv.backward()
        fp = x.grad.clone().detach()
        x.grad.data.zero_()
        gv = g(x)
        gv.backward()
        gp = x.grad.clone().detach()
        fpf = fp.flatten()
        gpf = gp.flatten()
        x.requires_grad_(False)
        sx = x + lr * (fp - (fpf @ gpf) / (gpf  @ gpf) * gp)
        sx = img_extra_constrain(sx)
        ssx = correcter(sx)
        if abs(g(ssx) - gdv) > 1.0001 * tol:
            if DEBUG:
                print('reason A')
            break
        x = ssx
        x = img_extra_constrain(x)
        if i > 0:
            if lr > 0 and fv - lastfv < eps:
                if DEBUG >= 4:
                    print('reason B')
                break
            if lr < 0 and fv - lastfv > - eps:
                if DEBUG >= 4:
                    print('reason B')
                break

        lastfv = fv
        i += 1
    return x


"""
Gradiant accend f with the constrain g(x) = gdv
"""
def optimize_eq_constrained_adam(
        x, f, g, gdv, iter,
        alpha, beta1=0.9, beta2=0.999, eps_adam=1e-8,
        tol=3e-4, lmd=1,
        eps=1e-5):
    ori_img = x.clone()
    x = x.clone()
    correcter = EqualConstrainCorrecter(g, gdv, lmd, tol)

    m = 0
    v = 0

    lastfv = 0
    last = x.clone()
    i = 0
    for i in range(iter):
        x.requires_grad_(True)
        print('\r{:10} '.format(i),  end='')

        if x.grad is not None:
            x.grad.data.zero_()
        fv = f(x)
        print('', fv.clone().detach().cpu().numpy(), end='')
        if DEBUG >= 4:
            print()
        fv.backward()
        fp = x.grad.clone().detach()
        x.grad.data.zero_()
        gv = g(x)
        gv.backward()
        gp = x.grad.clone().detach()

        m = beta1 * m + (1 - beta1) * fp
        v = beta2 * v + (1 - beta2) * (fp ** 2)
        mh = m / (1 - beta1 ** (i+1))
        vh = v / (1 - beta2 ** (i+1))
        odir = alpha * mh / (torch.sqrt(vh) + eps)

        fpf = odir.flatten()
        gpf = gp.flatten()

        x.requires_grad_(False)

        sx = x + (odir - (fpf @ gpf) / (gpf  @ gpf) * gp)
        sx = img_extra_constrain(sx)
        ssx = correcter(sx)
        if abs(g(ssx) - gdv) > 1.0001 * tol:
            print('reason A')
            break
        x = ssx
        x = img_extra_constrain(x)

        if torch.any(torch.isnan(x)):
            if DEBUG:
                print('Warning: NaN, the last one returned')
            return last
        if torch.any(torch.isnan(ssx)):
            if DEBUG:
                print('Warning: NaN, the current one returned')
            return x
        if alpha > 0 and fv - lastfv < 0:
            if DEBUG:
                print('Warning: learning to worse, last returned ')
            return last
        if i > 0:
            if alpha < 0 and fv - lastfv > - 0:
                if DEBUG >= 4:
                    print('reason B')
                break
            if alpha > 0 and fv - lastfv < eps or alpha < 0 and fv - lastfv > - eps:
                if DEBUG >= 4:
                    print('reason B')
                break

        lastfv = fv
        last = x
        i += 1
    return x

