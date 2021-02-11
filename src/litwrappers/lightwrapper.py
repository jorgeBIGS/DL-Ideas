import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl


class LitWrapper(pl.LightningModule):

    def __init__(self, model, optimizer, loss_function):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.model(x)
        return embedding

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        output = self.model(x)
        loss = self.loss_function(output, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer
        return optimizer
