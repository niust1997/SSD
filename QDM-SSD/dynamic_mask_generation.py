import torch
import pytorch_lightning as pl
from argparse import Namespace
from typing import Callable, Optional
from torch.optim.optimizer import Optimizer
import pdb
from ..utils import flatten_dict
import random as random


class System(pl.LightningModule):

    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            `{"interval": "batch", "scheduler": sched}` where `interval=="batch"`
            for batch-wise schedulers and `interval=="epoch"` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
    .. note:: By default, `training_step` (used by `pytorch-lightning` in the
        training loop) and `validation_step` (used for the validation loop)
        share `common_step`. If you want different behavior for the training
        loop and the validation loop, overwrite both `training_step` and
        `validation_step` instead.
    """

    def __init__(
        self,
        model,
        model_teacher,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.model_teacher = model_teacher
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        # hparams will be logged to Tensorboard as text variables.
        # torch doesn't support None in the summary writer for now, convert
        # None to strings temporarily.
        # See https://github.com/pytorch/pytorch/issues/33140
        self.hparams = Namespace(**self.config_to_hparams(config))

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note:: This is typically the method to overwrite when subclassing
            `System`. If the training and validation steps are somehow
            different (except for loss.backward() and optimzer.step()),
            the argument `train` can be used to switch behavior.
            Otherwise, `training_step` and `validation_step` can be overwriten.
        """
        
        inputs, targets = batch
        batch_size = targets.shape[0]
        loss_prob = []
        alpha = 1.0
        print(alpha)
        if torch.rand(1) >= (1 - alpha):
            for i in range(batch_size):
                cur_key = 0
                target_spk1 = targets[i, 0, :]
                target_spk2 = targets[i, 1, :]
                teacher_est_spk1 = self.model_teacher(target_spk1.unsqueeze(0))
                teacher_est_spk2 = self.model_teacher(target_spk2.unsqueeze(0))
                loss11 = self.loss_func(teacher_est_spk1[0,0,:].unsqueeze(0).unsqueeze(0), target_spk1.unsqueeze(0).unsqueeze(0), 0)[0]
                loss12 = self.loss_func(teacher_est_spk1[0,1,:].unsqueeze(0).unsqueeze(0), target_spk1.unsqueeze(0).unsqueeze(0), 0)[0]
                loss21 = self.loss_func(teacher_est_spk2[0,0,:].unsqueeze(0).unsqueeze(0), target_spk2.unsqueeze(0).unsqueeze(0), 0)[0]
                loss22 = self.loss_func(teacher_est_spk2[0,1,:].unsqueeze(0).unsqueeze(0), target_spk2.unsqueeze(0).unsqueeze(0), 0)[0]
                min_loss1, flag1 = self.min_loss_index(loss11, loss12)
                min_loss2, flag2 = self.min_loss_index(loss21, loss22)
                # cur_loss = max(min_loss1, min_loss2)
                # cur_max_loss = cur_loss
                # cur_min_loss = min(min_loss1, min_loss2)
                
                # -------- get p_i --------
                prob1 = self.loss_sig_prob(min_loss1)
                prob2 = self.loss_sig_prob(min_loss2)
                segment_length1 = int(prob1 * inputs.shape[-1])
                segment_length2 = int(prob2 * inputs.shape[-1])
                
                # -------- random start point --------
                start1 = random.randint(0, inputs.shape[-1] - segment_length1)
                start2 = random.randint(0, inputs.shape[-1] - segment_length2)
                
                fix_target1 = torch.zeros_like(target_spk1)
                fix_target1[start1: start1 + segment_length1] = target_spk1[start1: start1 + segment_length1]
                fix_target2 = torch.zeros_like(target_spk2)
                fix_target2[start2: start2 + segment_length2] = target_spk2[start2: start2 + segment_length2]
                
                targets[i, 0, :] = fix_target1
                targets[i, 1, :] = fix_target2
                inputs[i, :] = targets[i, 1, :] + targets[i, 0, :]
                
                if prob1 <= 0.1 and prob2 <= 0.1:
                    cur_key = torch.tensor(0).cuda()
                else:
                    cur_key = torch.tensor(1).cuda()
                if prob1 == 0 or prob2 == 0:
                    cur_key = torch.tensor(0).cuda()
                loss_prob.append(cur_key)
        else:
            loss_prob = [torch.tensor(1).cuda(),torch.tensor(1).cuda(),torch.tensor(1).cuda(),torch.tensor(1).cuda()]
            
            
        est_targets = self(inputs)
        loss, min_loss_idx = self.loss_func(est_targets, targets, loss_prob)
        final_loss = loss
        return final_loss
    
    
    # def loss_2_prob(self, loss): #
        # if loss <= -30:
            # prob = 1
        # elif loss <= -20:
            # prob = 0.1*((-loss-20)**0.5) + 0.6 # -20 ~ -30 ---> 0.6 - 0.9 
        # elif loss <= -10:
            # prob = 1.25e-7*(-loss**5) + 0.1 # -10 ~ -20 ---> 0.1125 - 0.5
        # elif loss < 0:
            # prob = 0.1 
        # else:
            # prob = 0
        # return prob
        
    # def check_start(self, input_target, input_target_sep, length, flag):
        # T = input_target.shape[0]
        # cycle_time = int((T - length) / 80)
        # if cycle_time <= 1 or cycle_time >= 100:
            # return random.randint(0, T - length)
        # best_start = 0
        # best_loss = 0
        # for i in range(cycle_time):
            # start = i * 80
            # end = start + length
            # sep = input_target_sep[0,flag - 1, start:end].unsqueeze(0).unsqueeze(0)
            # tgt = input_target[start:end].unsqueeze(0).unsqueeze(0)
            # loss = self.loss_func(sep, tgt, 0)[0]
            # if loss < best_loss:
                # best_loss = loss
                # best_start = start
            # print(loss)
        # pdb.set_trace()
        # return best_start
        
        
    # def check_start(self, input_target, input_target_sep, length, flag):
        # T = input_target.shape[0]
        # cycle_time = int((T - length) / 80)
        # if cycle_time <= 1 or cycle_time >= 100:
            # return random.randint(0, T - length)
        # best_start = 0
        # best_loss = 0
        # start_list = []
        # for i in range(cycle_time):
            # start = i * 80
            # end = start + length
            # sep = input_target_sep[0,flag - 1, start:end].unsqueeze(0).unsqueeze(0)
            # tgt = input_target[start:end].unsqueeze(0).unsqueeze(0)
            # loss = self.loss_func(sep, tgt, 0)[0]
            # if loss < best_loss:
                # best_loss = loss
                # best_start = start
            # if loss <= -15:
                # start_list.append(start)
            # # print(loss)
        # if len(start_list) > 1:
            # select_start = random.choice(start_list)
        # else:
            # select_start = best_start
            # # print(loss)
        # # pdb.set_trace()
        # return select_start
        
    
    # def check_start(self, input_target, input_target_sep, length, flag, cur_loss):
        # # pdb.set_trace()
        # T = input_target.shape[0]
        # cycle_time = int((T - length) / 80)
        # if cycle_time <= 1 or cycle_time >= 100:
            # return random.randint(0, T - length)
        # best_start = 0
        # best_loss = 0
        # start_list = []
        # for i in range(cycle_time):
            # start = i * 80
            # end = start + length
            # sep = input_target_sep[0,flag - 1, start:end].unsqueeze(0).unsqueeze(0)
            # tgt = input_target[start:end].unsqueeze(0).unsqueeze(0)
            # loss = self.loss_func(sep, tgt, 0)[0]
            # if loss < best_loss:
                # best_loss = loss
                # best_start = start
            # if loss <= -20:
                # start_list.append(start)
        # if len(start_list) > 1:
            # select_start = random.choice(start_list)
        # else:
            # select_start = best_start
            # # print(loss)
        # # pdb.set_trace()
        # return select_start
    
    
    
    
    
    def min_loss_index(self, loss1, loss2):
        flag = 0
        if loss1 <= loss2:
            min_loss = loss1
            flag = 1
        else:
            min_loss = loss2
            flag = 2
        return min_loss, flag
        
    
    def loss_2_prob26(self, loss): #
        if loss <= -30:
            prob = 1
        elif loss <= -20:
            prob = 0.1*((-loss-20)**0.5) + 0.6 # -20 ~ -30 ---> 0.6 - 0.9 
        elif loss <= -10:
            prob = 1.25e-7*(-loss**5) + 0.1 # -10 ~ -20 ---> 0.1125 - 0.5
        else:                               
            prob = 0
        return prob
    
    def loss_2_prob37(self, loss): # 
        if loss <= -30:
            prob = 1
        elif loss <= -20:
            prob = 0.1*((-loss-20)**0.5) + 0.6 # -20 ~ -30 ---> 0.6 - 0.9 
        elif loss <= -10:
            prob = 4e-6*((-loss-10)**5) + 0.1 # -10 ~ -19.9 ---> 0.1 - 0.6
        else:                               
            prob = 0
        return prob
        
    def loss_2_prob(self, loss): 
        if loss <= -30:
            prob = 1
        elif loss <= -20:
            prob = 0.1*((-loss-20)**0.5) + 0.6 # -20 ~ -30 ---> 0.6 - 0.9 
        elif loss <= -10:
            prob = 4e-3*((-loss-10)**2) + 0.1 # -10 ~ -19.9 ---> 0.1 - 0.6
        else:                               
            prob = 0
        return prob
        
        
    def loss_sig_prob(self, loss): 
        if loss <= -30:
            prob = 1
        elif loss <= -10:
            prob = max(1 / (1 + torch.exp(-(0.3 * ((-loss) - 20)))), 0.1)
        else:
            prob = 0
        return prob

        
    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'loss'``: loss

            ``'log'``: dict with tensorboard logs

        """
        loss = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def optimizer_step(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            for sched in self.scheduler:
                if isinstance(sched, dict) and sched["interval"] == "batch":
                    sched["scheduler"].step()  # call step on each batch scheduler
        super().optimizer_step(*args, **kwargs)

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'val_loss'``: loss
        """
        loss = self.common_step(batch, batch_nb, train=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """How to aggregate outputs of `validation_step` for logging.

        Args:
           outputs (list[dict]): List of validation losses, each with a
           ``'val_loss'`` key

        Returns:
            dict: Average loss

            ``'val_loss'``: Average loss on `outputs`

            ``'log'``: Tensorboard logs

            ``'progress_bar'``: Tensorboard logs
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""

        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            epoch_schedulers = []
            for sched in self.scheduler:
                if not isinstance(sched, dict):
                    epoch_schedulers.append(sched)
                else:
                    assert sched["interval"] in [
                        "batch",
                        "epoch",
                    ], "Scheduler interval should be either batch or epoch"
                    if sched["interval"] == "epoch":
                        epoch_schedulers.append(sched)
            return [self.optimizer], epoch_schedulers
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts `None` to
         ``'None'`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic

		
		
		
		
class System_multiInputs(pl.LightningModule):
    """Base class for deep learning systems.
    Contains a model, an optimizer, a loss function, training and validation
    dataloaders and learning rate scheduler.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            `{"interval": "batch", "scheduler": sched}` where `interval=="batch"`
            for batch-wise schedulers and `interval=="epoch"` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
    .. note:: By default, `training_step` (used by `pytorch-lightning` in the
        training loop) and `validation_step` (used for the validation loop)
        share `common_step`. If you want different behavior for the training
        loop and the validation loop, overwrite both `training_step` and
        `validation_step` instead.
    """

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
    ):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        config = {} if config is None else config
        self.config = config
        # hparams will be logged to Tensorboard as text variables.
        # torch doesn't support None in the summary writer for now, convert
        # None to strings temporarily.
        # See https://github.com/pytorch/pytorch/issues/33140
        self.hparams = Namespace(**self.config_to_hparams(config))

    def forward(self, *args, **kwargs):
        """Applies forward pass of the model.

        Returns:
            :class:`torch.Tensor`
        """
        return self.model(*args, **kwargs)

    def common_step(self, batch, batch_nb, train=True):
        """Common forward step between training and validation.

        The function of this method is to unpack the data given by the loader,
        forward the batch through the model and compute the loss.
        Pytorch-lightning handles all the rest.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.
            train (bool): Whether in training mode. Needed only if the training
                and validation steps are fundamentally different, otherwise,
                pytorch-lightning handles the usual differences.

        Returns:
            :class:`torch.Tensor` : The loss value on this batch.

        .. note:: This is typically the method to overwrite when subclassing
            `System`. If the training and validation steps are somehow
            different (except for loss.backward() and optimzer.step()),
            the argument `train` can be used to switch behavior.
            Otherwise, `training_step` and `validation_step` can be overwriten.
        """
		
        inputs,  inputs_embedding,  targets = batch
        est_targets = self(inputs,  inputs_embedding)
        loss = self.loss_func(est_targets, targets)
        return loss

    def training_step(self, batch, batch_nb):
        """Pass data through the model and compute the loss.

        Backprop is **not** performed (meaning PL will do it for you).

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'loss'``: loss
            ``'log'``: dict with tensorboard logs

        """
        loss = self.common_step(batch, batch_nb, train=True)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def optimizer_step(self, *args, **kwargs) -> None:
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            for sched in self.scheduler:
                if isinstance(sched, dict) and sched["interval"] == "batch":
                    sched["scheduler"].step()  # call step on each batch scheduler
        super().optimizer_step(*args, **kwargs)

    def validation_step(self, batch, batch_nb):
        """Need to overwrite PL validation_step to do validation.

        Args:
            batch: the object returned by the loader (a list of torch.Tensor
                in most cases) but can be something else.
            batch_nb (int): The number of the batch in the epoch.

        Returns:
            dict:

            ``'val_loss'``: loss
        """
        loss = self.common_step(batch, batch_nb, train=False)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        """How to aggregate outputs of `validation_step` for logging.

        Args:
           outputs (list[dict]): List of validation losses, each with a
           ``'val_loss'`` key

        Returns:
            dict: Average loss

            ``'val_loss'``: Average loss on `outputs`

            ``'log'``: Tensorboard logs

            ``'progress_bar'``: Tensorboard logs
        """
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs, "progress_bar": tensorboard_logs}

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is not None:
            if not isinstance(self.scheduler, (list, tuple)):
                self.scheduler = [self.scheduler]  # support multiple schedulers
            epoch_schedulers = []
            for sched in self.scheduler:
                if not isinstance(sched, dict):
                    epoch_schedulers.append(sched)
                else:
                    assert sched["interval"] in [
                        "batch",
                        "epoch",
                    ], "Scheduler interval should be either batch or epoch"
                    if sched["interval"] == "epoch":
                        epoch_schedulers.append(sched)
            return [self.optimizer], epoch_schedulers
        return self.optimizer

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def on_save_checkpoint(self, checkpoint):
        """ Overwrite if you want to save more things in the checkpoint."""
        checkpoint["training_config"] = self.config
        return checkpoint

    def on_batch_start(self, batch):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_batch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_start(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    def on_epoch_end(self):
        """ Overwrite if needed. Called by pytorch-lightning"""
        pass

    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts `None` to
         ``'None'`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.Tensor(v)
        return dic
		