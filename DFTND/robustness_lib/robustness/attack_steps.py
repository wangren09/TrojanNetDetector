import torch as ch
import numpy as np

class AttackerStep:

    def __init__(self, orig_input, eps, step_size):

        self.orig_input = orig_input 
        self.eps = eps
        self.step_size = step_size

    def project(self, x):

        raise NotImplementedError

    def make_step(self, g):

        raise NotImplementedError

    def random_perturb(self, x):

        raise NotImplementedError

### Instantiations of the AttackerStep class

# L-infinity threat model
class LinfStep(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = ch.clamp(diff, -self.eps, self.eps)
        return diff + self.orig_input

    def make_step(self, g):
        step = ch.sign(g) * self.step_size
        return step

    def random_perturb(self, x):
         return 2 * (ch.rand_like(x) - 0.5) * self.eps

class LinfStep1(AttackerStep):
    def project(self, a, gamma):
        diff = ch.sign(a) * ch.max(ch.abs(a) - self.step_size * gamma, ch.zeros_like(a))
        diff = ch.clamp(diff, 0, 1)
        return diff

    def make_step(self, g):
        step = ch.sign(g) * self.step_size
        return step

    def random_perturb(self, x):
         return 2 * (ch.rand_like(x) - 0.5) * self.eps

class LinfStep2(AttackerStep):
    def binary_search(self, a):

        a_ = a.detach().cpu().numpy()
        l = 0
        r = 10000

        mu = 0

        while r - l > 1e-6:
            mu = l + (r - l) / 2
            s = np.sum(np.maximum(0, a_ - mu))
            if s == 1:
                return mu
            elif s < 1:
                r = mu
            else:
                l = mu
        return mu

    def project(self, a):
        mu = self.binary_search(a)
        return ch.max(ch.zeros_like(a), a - mu)

    def make_step(self, g):
        step = ch.sign(g) * self.step_size
        return step

    def random_perturb(self, x):
         return 2 * (ch.rand_like(x) - 0.5) * self.eps

# L2 threat model
class L2Step(AttackerStep):
    def project(self, x):
        diff = x - self.orig_input
        diff = diff.renorm(p=2, dim=0, maxnorm=self.eps)
        return self.orig_input + diff

    def make_step(self, g):
        # Scale g so that each element of the batch is at least norm 1
        g_norm = ch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
        scaled_g = g / (g_norm + 1e-10)
        return scaled_g * self.step_size

    def random_perturb(self, x):
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.eps)

# Unconstrained threat model
class UnconstrainedStep(AttackerStep):
    def project(self, x):
        return x

    def make_step(self, g):
        return g * self.step_size

    def random_perturb(self, x):
        return (ch.rand_like(x) - 0.5).renorm(p=2, dim=1, maxnorm=self.step_size)
