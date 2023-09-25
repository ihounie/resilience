import torch
from absl import logging
from torch.nn.modules.loss import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.optim.adam import Adam
import torch.nn.functional as F

from lila.marglik import get_model_optimizer, get_scheduler, valid_performance


def uniform_aug(model,
             train_loader,
             valid_loader=None,
             likelihood='classification',
             weight_decay=1e-4,
             n_epochs=500,
             lr=1e-3,
             lr_min=None,
             optimizer='Adam',
             scheduler='exp',
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
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0

        # Pd training
        for X, y in train_loader:
            batch_size = len(y)
            optimizer.zero_grad()
            f = model(X)
            loss = criterion(f[:, 1], y)
            loss.backward()
            optimizer.step()
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
    return model, losses, valid_perfs, [epoch_perf]
