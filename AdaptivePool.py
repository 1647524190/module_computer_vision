import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptivePool(nn.Module):
    def __init__(self, kernel, stride, padding=0):
        """
        自适应池化
        Args:
            kernel (int): 池化窗口大小
            stride (int): 步长
            padding(int): 填充
        """
        super(AdaptivePool, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        max_pool = F.max_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)

        x = x.mean(dim=1, keepdim=True)
        weight = torch.sigmoid(x)

        out = weight * max_pool + (1 - weight) * avg_pool
        return out


# 测试
if __name__ == "__main__":
    x = torch.randn(1, 3, 8, 8)
    model = AdaptivePool(kernel=2, stride=2)
    output = model(x)
