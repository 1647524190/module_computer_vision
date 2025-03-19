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
        assert out_channels <= 2 * in_channels, "The number of output channels should be less than twice the number of input channels. "

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode = mode
        self.target_channels = self.out_channels if self.out_channels <= self.in_channels else self.out_channels - self.in_channels
        self.conv = Conv(self.target_channels, self.target_channels, 1, 1)

    def channel_pool(self, x):
        """
        通道池化操作
        Args:
            x (Tensor): 输入张量 (N, C, H, W)
        Returns:
            Tensor: 采样后的张量 (N, new_C, H, W)
        """
        stride = max(1, self.in_channels // self.target_channels)
        kernel = self.in_channels - (self.target_channels - 1) * stride

        # 变换形状并进行 `unfold`
        x_unfold = x.permute(0, 2, 3, 1).unfold(dimension=-1, size=kernel, step=stride)

        # 选择 `mean` 或 `max` 池化
        pooled = x_unfold.mean(dim=-1) if self.mode == "mean" else x_unfold.max(dim=-1)[0]
        pooled = pooled.permute(0, 3, 1, 2)
        # pooled = self.conv(pooled)

        return kernel, pooled

    def forward(self, x):
        kernel, pooled = self.channel_pool(x)

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

        return torch.cat(expanded_channels, dim=1)


if __name__ == "__main__":
    # 生成测试数据
    N, C, H, W = 2, 6, 4, 4  # batch=1, 通道数=5, 高度=4, 宽度=4
    x = torch.randn(N, C, H, W)  # 随机输入

    model = ChannelSample(in_channels=C, out_channels=2, mode="mean")
    out = model(x)
    print("输入 x:\n", x)
    print("\n输出形状:", out.shape)  # 期望 (1, 7, 4, 4)
    print(out)
