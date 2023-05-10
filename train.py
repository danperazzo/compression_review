#!/usr/bin/env python

"""
Builds a convolutional neural network on the fashion mnist data set.

Designed to show wandb integration with pytorch.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

import network
import dataset
import wandb
import os
import numpy as np

hyperparameter_defaults = dict(
    first_omega_0 = 30,
    hidden_omega_0 = 30,
    hidden_layers = 2,
    hidden_features = 256,
    learning_rate = 0.001,
    epochs = 100,
    )

wandb.init(config=hyperparameter_defaults, project="algelin_1rst_attempt",entity='alglin')
config = wandb.config



def main():
    
    model = network.Siren(in_features=1, out_features=1, first_omega_0=config.first_omega_0, 
                            hidden_omega_0= config.hidden_omega_0, hidden_features=config.hidden_features,
                             hidden_layers=config.hidden_layers, outermost_linear=True)

    fn = dataset.sines1
    train_signal_dataset = dataset.Func1DWrapper(range=(-0.5, 0.5),
                                         fn=fn,
                                         sampling_density=1000,
                                         train_every=1000/18)


    train_loader = torch.utils.data.DataLoader(dataset=train_signal_dataset,
                                               batch_size=1,
                                               shuffle=True)

    wandb.watch(model)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for epoch in range(config.epochs):
        
        for i, (coords, values) in enumerate(train_loader):

            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()

            # Forward pass to get output/logits
            outputs, _ = model(coords)

            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, values)

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()


        if epoch % 10 == 0:
            with torch.no_grad():
                # Forward pass to get output/logits
                outputs, _ = model(coords)

                # Calculate Loss: softmax --> cross entropy loss
                loss = criterion(outputs, values)
            
            metrics = {'loss': loss}
            wandb.log(metrics)
            # Print Loss
            print(f'Iteration: Loss: {loss} ')
    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "model.pt"))

if __name__ == '__main__':
   main()
