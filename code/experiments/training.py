import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from time import time
from torch import nn
from geoopt.optim import RiemannianAdam, RiemannianSGD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, config, train_loader, test_loader, path, seed,
                epochs=1000, save_model=False, measure_cartan=False):
    """Train a single model configuration."""
    torch.manual_seed(seed)
    start = time()

    lr = config['lr']
    wd = 1e-5
    nlayers = config['nlayers']
    optimizer_type = config.get('optimizer', 'adam').lower()

    criterion = nn.CrossEntropyLoss()
    rows = []

    if optimizer_type == "sgd":
        optimizer = RiemannianSGD(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = RiemannianAdam(model.parameters(), lr=lr, weight_decay=wd)

    pbar = tqdm(range(epochs), disable=os.getenv("DISABLE_TQDM", "false").lower() == "true")
    es = 15
    loss_buffer = np.inf * np.ones(es)
    best_model_index = es - 1

    for epoch in pbar:
        model.train()
        train_losses, train_accuracies = [], []

        for x_train, y_train in train_loader:
            x_train = x_train.flatten(start_dim=1).to(device)
            y_train = y_train.to(device)
            optimizer.zero_grad()

            outputs = model(x_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            acc = (torch.argmax(outputs, dim=-1) == y_train).float().mean().item()
            train_accuracies.append(acc)

        # Evaluate
        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test = x_test.flatten(start_dim=1).to(device)
                y_test = y_test.to(device)
                outputs = model(x_test)
                test_loss = criterion(outputs, y_test).item()
                test_acc = (torch.argmax(outputs, dim=-1) == y_test).float().mean().item()

        # Logging
        rows.append([
            config['mtype'].value,
            config['activation'].value,
            config['dataset'].value,
            test_acc,
            test_loss,
            np.mean(train_accuracies),
            np.mean(train_losses),
            epoch,
            lr,
            wd,
            config['optimizer'],
            config['nneurons'],
            nlayers,
            time() - start
        ])
        pbar.set_description(f"Loss {np.mean(train_losses):.4f}, Acc {test_acc:.4f}")

        # Early stopping
        loss_buffer[:-1] = loss_buffer[1:]
        loss_buffer[-1] = test_loss
        if np.all(loss_buffer[-1] < loss_buffer[:-1]):
            best_model_index = es - 1
            if save_model:
                torch.save(model.state_dict(), path / f"{seed}_checkpoint.pt")
        else:
            best_model_index -= 1
        if best_model_index <= 0:
            break

    df = pd.DataFrame(rows, columns=[
        'Model', 'Activation', 'Dataset', 'Test accuracy', 'Test loss',
        'Train accuracy', 'Train loss', 'Epoch', 'Learning rate',
        'Weight decay', 'Optimizer', 'Neurons', 'Nlayers', 'Time'
    ])
    df.to_csv(path / f"{seed}_data.csv", index=False)
