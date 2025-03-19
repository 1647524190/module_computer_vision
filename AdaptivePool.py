import torch
import torch.nn as nn
import torch.nn.functional as F
from test_gusssian import padding


class AdaptivePool(nn.Module):
    def __init__(self, kernel, stride, padding=0):
        """
        自适应池化：在 kernel_size x kernel_size 区域内动态选择最大池化或均值池化
        Args:
            kernel_size (int): 池化窗口大小
            stride (int): 步长
            padding(int): 填充
        """
        super(AdaptivePool, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

        self.weight_conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        # 计算 MaxPool 和 AvgPool
        max_pool = F.max_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel, stride=self.stride, padding=self.padding)

        x = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        weight = torch.sigmoid(self.weight_conv(x.mean(dim=1, keepdim=True)))

        out = weight * max_pool + (1 - weight) * avg_pool
        return out


# 测试
if __name__ == "__main__":
    x = torch.randn(1, 3, 8, 8)
    model = AdaptivePool(kernel=2, stride=2)
    output = model(x)

