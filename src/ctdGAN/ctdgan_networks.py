import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as f

from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import average_precision_score

# --------------------------------------------------------------------------------
# Critic/Discriminator
class ctdCritic(nn.Module):
    """Critic Discriminator for ctGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super().__init__()
        dim = input_dim * pac
        self._pac = pac
        self._pac_dim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [nn.Linear(dim, item), nn.LeakyReLU(0.2), nn.Dropout(0.5)]
            dim = item

        seq += [nn.Linear(dim, 1)]
        self._seq = nn.Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', lambda_=10):
        """Compute the gradient penalty. From the paper on improved Wasserstein GAN training."""
        alpha = torch.rand(real_data.size(0) // self._pac, 1, 1, device=device)
        alpha = alpha.repeat(1, self._pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, self._pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = (gradients_view ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, x):
        """Apply the Discriminator to the `input_`."""
        assert x.size()[0] % self._pac == 0
        return self._seq(x.view(-1, self._pac_dim))

# --------------------------------------------------------------------------------
# Generator
class Residual(nn.Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = nn.Linear(i, o)
        self.bn = nn.BatchNorm1d(o)
        self.relu = nn.ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class ctdGenerator(nn.Module):
    """Generator for ctGAN and ctdGAN"""

    def __init__(self, embedding_dim, architecture, data_dim):
        super().__init__()
        dim = embedding_dim
        seq = []
        for item in list(architecture):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(nn.Linear(dim, data_dim))
        self.seq = nn.Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data

# --------------------------------------------------------------------------------
# Classifier network
class ctdClassifier(nn.Module):
    """
    ctdClassifier
    """
    # This architecture is for ctdGAN_cls2 and ctdGAN_cls2clu
    def __init__(self, input_dim, num_classes, hidden_dims=(128, 256, 256, 128), dropout=0.2, temperature=1.0):
        super().__init__()

        self.temperature = temperature

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[2], hidden_dims[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_dims[3], num_classes)
        )

    def forward(self, x):
        logits = self.net(x)
        # temperature scaling
        logits = logits / self.temperature

        return logits


# =========================================================
# Training Function
# =========================================================
def train_classifier(x_tr, y_tr, x_val, y_val, input_dim, num_classes, hidden_dims=(128, 256, 256, 128),
                     batch_size=64, epochs=30, lr=1e-3, weight_decay=1e-4, patience=5, device="cuda", random_state=0):
    """
    Train the classifier network.

    Args:
        x_tr: the training data.
        y_tr: the training labels.
        x_val: the validation data.
        y_val: the validation labels.
        input_dim: the input dimension.
        num_classes: the number of classes.
        hidden_dims: the number and size of the hidden layers.
        batch_size: the batch size.
        epochs: the number of epochs.
        lr: the learning rate.
        weight_decay: the weight decay.
        patience: the number of epochs without improvement.
        device:
        random_state: the random seed.
    :return:
        The trained classifier network.
    """
    # dataloaders
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

    # class weights
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_tr.cpu().numpy()),
                                         y=y_tr.cpu().numpy())
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # model
    model = ctdClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims,
                          dropout=0.2, temperature=1.0).to(device)

    # optimizer + loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # early stopping
    best_model = None
    best_val_loss = float("inf")
    patience_counter = 0

    # training loop
    for epoch in range(epochs):
        # train
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0

        all_probs, all_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                logits = model(xb)
                loss = criterion(logits, yb)
                val_loss += loss.item()

                probs = f.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        val_loss /= len(val_loader)

        #all_probs = np.concatenate(all_probs)
        #all_targets = np.concatenate(all_targets)

        # PR-AUC
        #if num_classes == 2:
        #    pr_auc = average_precision_score(all_targets, all_probs[:, 1])
        #else:
        #    pr_auc = average_precision_score(all_targets, all_probs, average="macro")

        #print(f"Epoch {epoch+1:03d} | " f"Train Loss: {train_loss:.4f} | " f"Val Loss: {val_loss:.4f} | " f"PR-AUC: {pr_auc:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            # print("Early stopping triggered at epoch ", epoch)
            break

    # load best model
    model.load_state_dict(best_model)
    return model
