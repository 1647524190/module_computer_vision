import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePool(nn.Module):
    def __init__(self, kernel, stride, padding=0):
        super(AdaptivePool, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=kernel, stride=stride, padding=padding)
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel, stride=stride, padding=padding)

    def forward(self, x):
        max_pool = self.max_pool(x)
        avg_pool = self.avg_pool(x)

        attn = (max_pool * avg_pool).mean(dim=1, keepdims=True)
        attn = torch.sigmoid(attn)

        out = attn * avg_pool + (1 - attn) * max_pool
        return out


# 测试
if __name__ == "__main__":
    x = torch.randn(1, 2, 4, 4)
    model = AdaptivePool(kernel=2, stride=2)
    output = model(x)
