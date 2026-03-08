import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================
# Masked Linear Layer
# ===============================
    
class MaskedLinear(nn.Module):
    def __init__(self, in_features, out_features, mask, activation="relu"):
        super().__init__()

        self.register_buffer("mask", mask)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.reset_parameters(activation)

    def reset_parameters(self, activation):
        fan_in = self.mask.sum(dim=1)
        fan_in = torch.clamp(fan_in, min=1)

        if activation.lower() == "relu":
            std = torch.sqrt(2.0 / fan_in)
        else: 
            std = torch.sqrt(1.0 / fan_in)

        with torch.no_grad():
            for i in range(self.weight.shape[0]):
                self.weight[i].normal_(0, std[i])
    def forward(self, x):
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight, self.bias)


# ===============================
# Flexible Partial Network
# ===============================


class PartialNet(nn.Module):
    def __init__(self, input_dim, hidden1, hidden2,
                 fc_layers, mask1, mask2,
                 activation="relu", use_layernorm=True, dropout=0.0):

        super().__init__()

        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError("Unsupported activation")

        self.masked1 = MaskedLinear(input_dim, hidden1, mask1, activation)
        self.masked2 = MaskedLinear(hidden1, hidden2, mask2, activation)

        self.norm1 = nn.LayerNorm(hidden1) if use_layernorm else nn.Identity()
        self.norm2 = nn.LayerNorm(hidden2) if use_layernorm else nn.Identity()

        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        layers = []
        prev_dim = hidden2

        for dim in fc_layers:
            linear = nn.Linear(prev_dim, dim)
            if activation in ["relu","gelu"]:
                nn.init.kaiming_normal_(linear.weight, nonlinearity="relu")
            elif activation == 'sigmoid':
                nn.init.kaiming_normal_(linear.weight, nonlinearity="linear")
            
            nn.init.zeros_(linear.bias)
            
            layers.append(linear)
            layers.append(nn.LayerNorm(dim) if use_layernorm else nn.Identity())
            layers.append(self.activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = dim

        final = nn.Linear(prev_dim, 1)
        nn.init.xavier_normal_(final.weight)

        layers.append(final)

        self.fc_stack = nn.Sequential(*layers)

    def forward(self, x):
        x = self.masked1(x)
        x = self.norm1(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)

        x = self.masked2(x)
        x = self.norm2(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)

        x = self.fc_stack(x)
        return x

# ===============================
# Early Stopping epochs
# ===============================

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, max_delta =1e-3):
        self.patience = patience
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        elif (val_loss > self.best_loss + self.max_delta) and (abs(val_loss - loss) > 3e-2):
            self.counter += 1

            if self.counter >= self.patience:
                self.early_stop = True
