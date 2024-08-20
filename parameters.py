import torch


class Parameters(torch.nn.Module):
    def __init__(self, batch_size, size1, size2):
        super(Parameters, self).__init__()
        self.data = .5 * torch.randn(batch_size, 256, size1 // 16, size2 // 16)
        self.data = torch.nn.Parameter(torch.sin(self.data))

    def forward(self):
        return self.data
