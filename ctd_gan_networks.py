import torch
import torch.nn as nn


# import DeepCoreML.paths as paths
class Critic(nn.Module):
    """Discriminator for ctGAN."""

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
        """Compute the gradient penalty. From the paper on improved WGAN training."""
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
        """Apply the Discriminator to the `input`."""
        assert x.size()[0] % self._pac == 0
        return self._seq(x.view(-1, self._pac_dim))


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


class Generator(nn.Module):
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
