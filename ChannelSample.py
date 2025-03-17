import torch
import torch.nn as nn
from Conv import Conv  # 确保 `Conv` 这个模块存在


class ChannelSample(nn.Module):
    def __init__(self, in_channels, out_channels, mode="mean"):
        """
        在通道维度上进行采样（池化）或插值扩展
        Args:
            in_channels (int): 输入通道数
            out_channels (int, optional): 输出通道数
            mode (str): "mean"（均值池化）或 "max"（最大池化）
        """
        super(ChannelSample, self).__init__()
        assert mode in ["mean", "max"], "mode must be 'mean' or 'max'"

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode

        # # 计算是否需要额外的卷积层
        # if self.out_channels <= self.in_channels:
        #     self.conv = Conv(self.out_channels, self.out_channels, 1, 1)
        # else:
        #     self.conv = Conv(self.out_channels - self.in_channels, self.out_channels - self.in_channels, 1, 1)

    def compute_kernel_stride(self, in_channels, out_channels):
        """
        计算通道池化 kernel_size 和 stride
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
        Returns:
            kernel (int), stride (int)
        """
        # 获取插值通道数
        target_channels = out_channels if out_channels <= in_channels else out_channels - in_channels

        stride = max(1, in_channels // target_channels)
        kernel = in_channels - (target_channels - 1) * stride

        return kernel, stride

    def unfold_and_pool(self, x, kernel, stride):
        """
        通道池化操作
        Args:
            x (Tensor): 输入张量 (N, C, H, W)
            kernel: 通道池化核
            stride：通道池化步长
        Returns:
            Tensor: 采样后的张量 (N, new_C, H, W)
        """

        # 变换形状并进行 `unfold`
        x_unfold = x.permute(0, 2, 3, 1).unfold(dimension=-1, size=kernel, step=stride)

        # 选择 `mean` 或 `max` 池化
        pooled = x_unfold.mean(dim=-1) if self.mode == "mean" else x_unfold.max(dim=-1)[0]
        pooled = pooled.permute(0, 3, 1, 2)
        # pooled = self.conv(pooled)

        return pooled

    def forward(self, x):
        kernel, stride = self.compute_kernel_stride(self.in_channels, self.out_channels)

        pooled = self.unfold_and_pool(x, kernel, stride)

        if self.out_channels <= self.in_channels:
            return pooled

        expanded_channels = []
        orig_channels = x  # (N, C, H, W)
        pooled_idx, i = 0, 0

        while i < self.in_channels:
            expanded_channels.append(orig_channels[:, i:i + kernel, :, :])
            i += kernel
            if pooled_idx < (self.out_channels - self.in_channels):
                expanded_channels.append(pooled[:, pooled_idx:pooled_idx + 1, :, :])
                pooled_idx += 1

        return torch.cat(expanded_channels, dim=1)  # 拼接插值后的通道


if __name__ == "__main__":
    # 生成测试数据
    N, C, H, W = 1, 4, 4, 4  # batch=1, 通道数=5, 高度=4, 宽度=4
    x = torch.randn(N, C, H, W)  # 随机输入
    print("输入 x:\n", x)

    model = ChannelSample(in_channels=C, out_channels=2, mode="mean")
    out = model(x)
    print("\n输出形状:", out.shape)  # 期望 (1, 7, 4, 4)
    print(out)
