import torch

# Source: https://lightning.ai/lightning-ai/studios/code-lora-from-scratch?view=public&section=featured


class RSLoRALayer(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: int):
        """Initialize a LoRA (Low-Rank Adaptation) layer.

        Parameters
        ----------
        in_dim : int
            Input dimension size of the layer. This should match the dimension of
            the input tensor that will be passed to the forward method.
        out_dim : int
            Output dimension size of the layer. The layer will project the input
            to this dimension through the low-rank decomposition.
        rank : int
            Rank of the low-rank decomposition matrices. This determines the compression
            ratio and number of trainable parameters. A lower rank means fewer parameters
            but potentially less model capacity.
        alpha : int
            Scaling factor for the LoRA update. This hyperparameter scales the
            contribution of the LoRA update to the output. Higher values mean
            stronger LoRA adaptation effects.

        """
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.A = torch.nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha
        self.rank = rank

    def forward(self, x: torch.Tensor):
        x = (self.alpha / (self.rank**0.5)) * self.B(self.A(x))
        return x


class LinearWithRSLoRA(torch.nn.Module):
    def __init__(self, linear: torch.nn.Linear, rank: int, alpha: int):
        """Initialize a Linear layer with LoRA adaptation.

        Parameters
        ----------
        linear : torch.nn.Linear
            The base linear layer to adapt with LoRA. This layer's weights will
            remain frozen while the LoRA parameters are trained.
        rank : int
            Rank of the low-rank decomposition matrices. This determines the compression
            ratio and number of trainable parameters. A lower rank means fewer parameters
            but potentially less model capacity.
        alpha : int
            Scaling factor for the LoRA update. This hyperparameter scales the
            contribution of the LoRA update to the output. Higher values mean
            stronger LoRA adaptation effects.
        """
        super().__init__()
        self.linear = linear
        self.lora = RSLoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x: torch.Tensor):
        return self.linear(x) + self.lora(x)
