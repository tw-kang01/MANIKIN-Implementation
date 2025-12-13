"""
MANIKIN Model Base Class
Adapted from EgoPoser/models/model_base.py
"""

import os
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


class ModelBase():
    """Base class for MANIKIN training wrapper"""

    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path'].get('models', 'Manikin/outputs/models')
        self.device = torch.device('cuda' if opt.get('gpu_ids') is not None else 'cpu')
        self.is_train = opt.get('is_train', True)
        self.schedulers = []

    # ----------------------------------------
    # Preparation before training with data
    # ----------------------------------------

    def init_train(self):
        pass

    def load(self, test=False):
        pass

    def save(self, label):
        pass

    def define_loss(self):
        pass

    def define_optimizer(self):
        pass

    def define_scheduler(self):
        pass

    # ----------------------------------------
    # Optimization during training with data
    # ----------------------------------------

    def feed_data(self, data):
        pass

    def optimize_parameters(self, current_step):
        pass

    def test(self):
        pass

    def current_log(self):
        pass

    def current_visuals(self):
        pass

    def current_losses(self):
        pass

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step()

    def current_learning_rate(self):
        if len(self.schedulers) > 0:
            return self.schedulers[0].get_last_lr()[0]
        return 0.0

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    # ----------------------------------------
    # Information of network
    # ----------------------------------------

    def print_network(self):
        pass

    def info_network(self):
        if hasattr(self, 'net'):
            return self.describe_network(self.net)
        return "No network defined"

    def print_params(self):
        pass

    def info_params(self):
        if hasattr(self, 'net'):
            return self.describe_params(self.net)
        return "No network defined"

    def get_bare_model(self, network):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(network, (DataParallel, DistributedDataParallel)):
            network = network.module
        return network

    def model_to_device(self, network):
        """Model to device. Optionally wrap with DataParallel."""
        network = network.to(self.device)
        if self.opt.get('dist', False):
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            network = DistributedDataParallel(
                network,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=find_unused_parameters
            )
        elif self.opt.get('use_dataparallel', False):
            network = DataParallel(network)
        return network

    # ----------------------------------------
    # Network name and number of parameters
    # ----------------------------------------
    def describe_network(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += 'Networks name: {}'.format(network.__class__.__name__) + '\n'
        msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), network.parameters()))) + '\n'
        msg += 'Net structure:\n{}'.format(str(network)) + '\n'
        return msg

    # ----------------------------------------
    # Parameters description
    # ----------------------------------------
    def describe_params(self, network):
        network = self.get_bare_model(network)
        msg = '\n'
        msg += ' | {:^6s} | {:^6s} | {:^6s} | {:^6s} || {:<20s}'.format(
            'mean', 'min', 'max', 'std', 'param_name') + '\n'
        for name, param in network.state_dict().items():
            if 'num_batches_tracked' not in name:
                v = param.data.clone().float()
                msg += ' | {:>6.3f} | {:>6.3f} | {:>6.3f} | {:>6.3f} | {} || {:s}'.format(
                    v.mean(), v.min(), v.max(), v.std(), v.shape, name) + '\n'
        return msg

    # ----------------------------------------
    # Save/Load network
    # ----------------------------------------
    def save_network(self, save_dir, network, iter_label):
        save_filename = '{}.pth'.format(iter_label)
        save_path = os.path.join(save_dir, save_filename)
        network = self.get_bare_model(network)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    def load_network(self, load_path, network, strict=True, param_key='params'):
        network = self.get_bare_model(network)
        state_dict = torch.load(load_path, map_location=self.device)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        network.load_state_dict(state_dict, strict=strict)

    # ----------------------------------------
    # Save/Load optimizer
    # ----------------------------------------
    def save_optimizer(self, save_dir, optimizer, optimizer_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, optimizer_label)
        save_path = os.path.join(save_dir, save_filename)
        torch.save(optimizer.state_dict(), save_path)

    def load_optimizer(self, load_path, optimizer):
        optimizer.load_state_dict(
            torch.load(load_path, map_location=self.device)
        )
