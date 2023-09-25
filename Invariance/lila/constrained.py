import torch
from absl import logging
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim.adam import Adam
import torch.nn.functional as F

from lila.marglik import get_model_optimizer, get_scheduler, valid_performance


def primal_dual(model,
             train_loader,
             valid_loader=None,
             likelihood='classification',
             weight_decay=1e-4,
             epsilon=0.1,
             lr_dual=0.01,
             n_epochs=500,
             lr=1e-3,
             lr_min=None,
             optimizer='Adam',
             scheduler='exp',
             keep_all=False,
             augmenter=None):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    if likelihood == 'classification':
        criterion = CrossEntropyLoss()
    elif likelihood == 'regression':
        criterion = MSELoss()
    else:
        raise ValueError(f'Invalid likelihood: {likelihood}')

    losses = list()
    valid_perfs = list()
    dual_var = 0
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # Pd training
        for X, y in train_loader:
            batch_size = len(y)
            optimizer.zero_grad()
            f = model(X)
            clean_loss = criterion(f[:, 0], y)
            augmented = f[:,1:]
            aug_loss = torch.empty((augmented.shape[:-1]))
            for i in range(augmented.shape[1]):
                aug_loss[:, i] = criterion(augmented[:, i], y)
            #############
            # MH-Sampling
            #############
            # First step
            if keep_all==True:#We keep the whole chain
                ones = torch.ones_like(clean_loss)
                # More steps
                for step in range(2, aug_loss.shape[1]):
                    acceptance_ratio = torch.minimum(torch.nan_to_num(aug_loss[:,step] / mh_loss), ones)
                    acceptance_ratio =  acceptance_ratio * (acceptance_ratio > 0)
                    accepted = torch.bernoulli(acceptance_ratio).bool()
                    aug_loss[accepted == False, step] = aug_loss[accepted == False,step-1]
                mh_loss = aug_loss
            else:# We only keep the last sample in the chain
                ones = torch.ones_like(clean_loss)
                mh_loss = aug_loss[:,0]
                # More steps
                for step in range(2, aug_loss.shape[1]):
                    acceptance_ratio = torch.minimum(torch.nan_to_num(aug_loss[:,step] / mh_loss), ones)
                    acceptance_ratio =  acceptance_ratio * (acceptance_ratio > 0)
                    accepted = torch.bernoulli(acceptance_ratio).bool()
                    mh_loss[accepted] = aug_loss[accepted,step]
            mh_loss = torch.mean(mh_loss)
            loss = clean_loss + dual_var*mh_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                slack = mh_loss-epsilon
                dual_var = F.relu(dual_var+lr_dual*slack)
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f[:,0].detach(), dim=-1) == y).item() / N
            scheduler.step()
        losses.append(epoch_loss)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method='avgfunc', device=device)
                valid_perfs.append(valid_perf)
                logging.info(f'Constrained[epoch={epoch}]: validation performance {valid_perf*100:.2f}.%')
    return model, losses, valid_perfs, [epoch_perf], dual_var

def primal_dual_all(model,
             train_loader,
             valid_loader=None,
             likelihood='classification',
             weight_decay=1e-4,
             epsilon=0.1,
             lr_dual=0.01,
             n_epochs=500,
             lr=1e-3,
             lr_min=None,
             optimizer='Adam',
             scheduler='exp',
             keep_all=False,
             augmenter=None):
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr, weight_decay)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)

    n_steps = augmenter.n_samples

    if likelihood == 'classification':
        criterion = CrossEntropyLoss()
    elif likelihood == 'regression':
        criterion = MSELoss()
    else:
        raise ValueError(f'Invalid likelihood: {likelihood}')

    losses = list()
    valid_perfs = list()
    dual_var = torch.zeros(3).to(device)
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # Pd training
        for X, y in train_loader:
            batch_size = len(y)
            optimizer.zero_grad()
            augmenter.mode="identity"
            f = model(X.unsqueeze(1)).squeeze(1)
            clean_loss = criterion(f, y)
            aug_loss = torch.zeros(batch_size, 3, n_steps).to(device)
            for t_num, transform in enumerate(["rotation","translation", "scale"]):
                augmenter.mode = transform
                X_aug = augmenter(X)
                augmented = model(X_aug)
                for step in range(n_steps):
                    aug_loss[:, t_num, step] = criterion(augmented[:, step, :], y)
            #############
            # MH-Sampling
            #############
            # First step
            if keep_all==True:#We keep the whole chain
                raise NotImplementedError
            else:# We only keep the last sample in the chain
                ones = torch.ones_like(clean_loss)
                mh_loss = aug_loss[:,:, 0]
                # More steps
                for transform in range(3):
                    for step in range(1, n_steps):
                        acceptance_ratio = torch.minimum(torch.nan_to_num(aug_loss[:,transform,step] / mh_loss[:, transform]), ones)
                        acceptance_ratio =  acceptance_ratio * (acceptance_ratio > 0)
                        accepted = torch.bernoulli(acceptance_ratio).bool()
                        mh_loss[accepted, transform] = aug_loss[accepted,transform,step]
                mh_loss = torch.mean(mh_loss, dim=0)
            loss = clean_loss + torch.sum(dual_var*mh_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                slack = mh_loss-epsilon
                dual_var = F.relu(dual_var+lr_dual*slack)
            epoch_loss += loss.cpu().item() / len(train_loader)
            if likelihood == 'regression':
                epoch_perf += (f.detach() - y).square().sum() / N
            else:
                epoch_perf += torch.sum(torch.argmax(f[:,0].detach(), dim=-1) == y).item() / N
            scheduler.step()
        named_dual = {"dual_rotation":dual_var[0].detach().item(), "dual_translation":dual_var[1].detach().item(), "dual_scale":dual_var.detach()[2].detach().item()}
        for name, value in named_dual.items():
            print(name, value)
        scheduler.step()
        named_slack = {"slack_rotation":slack[0].detach().item(), "slack_translation":slack[1].detach().item(), "slack_scale":slack.detach()[2].detach().item()}
        for name, value in named_slack.items():
            print(name, value)
        losses.append(epoch_loss)

        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                valid_perf = valid_performance(model, valid_loader, likelihood, method='avgfunc', device=device)
                valid_perfs.append(valid_perf)
                logging.info(f'Constrained[epoch={epoch}]: validation performance {valid_perf*100:.2f}.%')
    return model, losses, valid_perfs, [epoch_perf], dual_var
