import torch as ch
import dill
import os
if int(os.environ.get("NOTEBOOK_MODE", 0)) == 1:
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm

from . import helpers
from . import attack_steps
import numpy as np

STEPS = {
    'inf': attack_steps.LinfStep,
    '2': attack_steps.L2Step,
    'unconstrained': attack_steps.UnconstrainedStep
}

class Attacker(ch.nn.Module):

    def __init__(self, model, dataset):

        super(Attacker, self).__init__()
        self.normalize = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model

    def forward(self, x, target, *_, constraint, eps, step_size, iterations, criterion,
                random_start=False, random_restarts=False, do_tqdm=False,
                targeted=False, custom_loss=None, should_normalize=True, 
                orig_input=None, use_best=False, gamma=0.01, sigma=0.000001):

        
        # Can provide a different input to make the feasible set around
        # instead of the initial point

        if orig_input is None: orig_input = x.detach()
        orig_input = orig_input.cuda()

        # Multiplier for gradient ascent [untargeted] or descent [targeted]
        m = -1 if targeted else 1

        # Initialize step class
        step = STEPS[constraint](eps=eps, orig_input=orig_input, step_size=step_size)

        def calc_loss(inp, target):

            if should_normalize:
                inp = self.normalize(inp)
            output = self.model(inp)
            if custom_loss:
                return custom_loss(self.model, inp, target)

            return criterion(output, target), output


        def get_adv_examples(x):

            pert = ch.empty(x.shape).normal_(mean=0,std=sigma).cuda()
            #print(ch.max(pert).item())
            # Random start (to escape certain types of gradient masking)
            if random_start:
                x = ch.clamp(x + step.random_perturb(x), 0, 1)

            iterator = range(iterations)
            if do_tqdm: iterator = tqdm(iterator)

            # Keep track of the "best" (worst-case) loss and its
            # corresponding input
            best_loss = None
            best_x = None

            # A function that updates the best loss and best input
            def replace_best(loss, bloss, x, bx):
                if bloss is None:
                    bx = x.clone().detach()
                    bloss = losses.clone().detach()
                else:
                    replace = m * bloss < m * loss
                    bx[replace] = x[replace].clone().detach()
                    bloss[replace] = loss[replace]

                return bloss, bx

            delta = ch.zeros_like(x, requires_grad=True).requires_grad_(True)
            #print(delta.shape)
            M = 0.5 * ch.zeros_like(x, requires_grad=True).requires_grad_(True)
            #print(M.shape)
            
            #weight method
            W = ch.ones(2048, requires_grad=True).requires_grad_(True).cuda() / 2048
            
            #If using the refine fix method, please comment the weight method and uncomment the fix method
            #fix method
#             W = ch.zeros(2048, requires_grad=True).requires_grad_(True).cuda()
#             W[1858] = 1

            #
            x0 = x

            step_d = STEPS[constraint](eps=eps, orig_input=delta, step_size=step_size)
            step_m = attack_steps.LinfStep1(eps=eps, orig_input=M, step_size=step_size)
            #weight method
            step_w = attack_steps.LinfStep2(eps=eps, orig_input=W, step_size=step_size)
            #
            for _ in iterator:
                delta = delta.clone().detach().requires_grad_(True)
                M = M.detach().requires_grad_(True)
                
                #weight method
                W = W.detach().requires_grad_(True)
                #
                x = (1 - M) * (x0 + pert) + M * delta
                
                
                #x = x + pert

                x = ch.clamp(x, 0, 1)
                losses, out = calc_loss(ch.clamp(x, 0, 1), target)
                
#                 W = ch.zeros_like(losses, requires_grad=True).requires_grad_(True)
#                 maxp = ch.argmax(losses, 1)
#                 for iii in range(10):
#                     W[iii][maxp[iii]] = 1
                
                W1 = W.unsqueeze(0).expand(10, -1)

                #W1 = W
                losses = losses * W1

                loss = ch.mean(ch.sum(losses,dim=1)) - gamma * ch.norm(M, 1)
                #weight method
                grad_d, grad_m, grad_w = ch.autograd.grad(loss, [delta, M, W])
                #fix method
#                 grad_d, grad_m = ch.autograd.grad(loss, [delta, M])
                #


                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    delta = step_d.make_step(grad_d) * m + delta
                    delta = step_d.project(delta)
                    
#                     #additional inf_norm constraint (for clean label attack)
#                     max_d = x0+20.0/255#ch.min(20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), eps)
#                     min_d = x0-20.0/255#ch.max(-20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), 0)
#                     delta = ch.where(delta > min_d, min_d, delta)
#                     delta = ch.where(delta < max_d, max_d, delta)

                    M = step_m.make_step(grad_m) * m + M
                    M = step_m.project(M, gamma)

                    #weight method
                    W = step_w.make_step(grad_w) * m + W
                    W = step_w.project(W)
                    #

                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))
            
            #refine using the average
            delta = delta.mean(0)
            delta = delta.unsqueeze(0).expand(10, -1, -1, -1).requires_grad_(True)
            M = M.mean(0)
            M = M.unsqueeze(0).expand(10, -1, -1, -1).requires_grad_(True)
            #weight method
            W = W.requires_grad_(True)
            
            step_d = STEPS[constraint](eps=eps, orig_input=delta, step_size=step_size)
            step_m = attack_steps.LinfStep1(eps=eps, orig_input=M, step_size=step_size)
            #weight method
            step_w = attack_steps.LinfStep2(eps=eps, orig_input=W, step_size=step_size)
            #
            for _ in iterator:
                delta = delta.clone().detach().requires_grad_(True)
                M = M.detach().requires_grad_(True)
                
                #weight method
                W = W.detach().requires_grad_(True)
                #
                x = (1 - M) * (x0 + pert) + M * delta
                
                
                #x = x + pert

                x = ch.clamp(x, 0, 1)
                losses, out = calc_loss(ch.clamp(x, 0, 1), target)
                W1 = W.unsqueeze(0).expand(10, -1)
                
#                 W = ch.zeros_like(losses, requires_grad=True).requires_grad_(True)
#                 maxp = ch.argmax(losses, 1)
#                 for iii in range(10):
#                     W[iii][maxp[iii]] = 1.0
#                 W1 = W
                
                losses = losses * W1

                loss = ch.mean(ch.sum(losses,dim=1)) - gamma * ch.norm(M, 1)
                #weight method
                grad_d, grad_m, grad_w = ch.autograd.grad(loss, [delta, M, W])
                #fix method
#                 grad_d, grad_m = ch.autograd.grad(loss, [delta, M])
                #
                with ch.no_grad():
                    args = [losses, best_loss, x, best_x]
                    best_loss, best_x = replace_best(*args) if use_best else (losses, x)

                    delta = step_d.make_step(grad_d) * m + delta
                    delta = step_d.project(delta)
                    
#                     #additional inf_norm constraint
#                     max_d = x0+20.0/255#ch.min(20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), eps)
#                     min_d = x0-20.0/255#ch.max(-20/M.cpu().detach().numpy() + x0.cpu().detach().numpy(), 0)
#                     delta = ch.where(delta > min_d, min_d, delta)
#                     delta = ch.where(delta < max_d, max_d, delta)
                    

                    M = step_m.make_step(grad_m) * m + M
                    M = step_m.project(M, gamma)

                    #weight method
                    W = step_w.make_step(grad_w) * m + W
                    W = step_w.project(W)
                    #

                    if do_tqdm: iterator.set_description("Current loss: {l}".format(l=loss))
            
            
            
            x = (1 - M) * x0 + M * delta
#             WW = [(j,i) for i, j in enumerate(W.data.cpu().numpy()) if j>0]
#             WW = sorted(WW)
#             print(len(WW))
#             print(WW[-100:])
            loss_ave = losses.mean(0)
            #print(losses.shape)
            #WW = [j for i, j in enumerate(loss_ave.data.cpu().numpy()) if j>0]
            
#             WW = [(j,i) for i, j in enumerate(losses[1].data.cpu().numpy())]
#             WW = sorted(WW)
#             print(WW[-100:])
            # Save computation (don't compute last loss) if not use_best
            if not use_best: return ch.clamp(x,0,1).clone().detach()

            losses, _ = calc_loss(x, target)
            args = [losses, best_loss, x, best_x]
            best_loss, best_x = replace_best(*args)
            return best_x



        # Random restarts: repeat the attack and find the worst-case
        # example for each input in the batch
        if random_restarts:
            to_ret = None

            orig_cpy = x.clone().detach()
            for _ in range(random_restarts):
                adv, WW = get_adv_examples(orig_cpy)
                #print(WW)

                if to_ret is None:
                    to_ret = adv.detach()

                _, output = calc_loss(adv, target)
                corr, = helpers.accuracy(output, target, topk=(1,), exact=True)
                corr = corr.byte()
                misclass = ~corr
                to_ret[misclass] = adv[misclass]

            adv_ret = to_ret
        else:
            adv_ret = get_adv_examples(x)

        return adv_ret

class AttackerModel(ch.nn.Module):
    def __init__(self, model, dataset):
        super(AttackerModel, self).__init__()
        self.normalizer = helpers.InputNormalize(dataset.mean, dataset.std)
        self.model = model
        self.attacker = Attacker(model, dataset)

    def forward(self, inp, target=None, make_adv=False, with_latent=False,
                    fake_relu=False, with_image=True, **attacker_kwargs):
        if make_adv:
            assert target is not None
            prev_training = bool(self.training)
            self.eval()
            adv = self.attacker(inp, target, **attacker_kwargs)
            if prev_training:
                self.train()

            inp = adv

        if with_image:
            normalized_inp = self.normalizer(inp)
            output = self.model(normalized_inp, with_latent=with_latent,
                                                    fake_relu=fake_relu)
        else:
            output = None

        return (output, inp)
