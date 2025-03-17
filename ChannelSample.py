import torch
import torch.nn as nn
from Conv import Conv


class ChannelSample(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=None, stride=None, mode="mean"):
        """
        在通道维度上进行采样（池化）
        Args:
            out_channels (int, optional): 目标输出通道数 (如果不指定 kernel_size & stride)
            kernel_size (int, optional): 滑动窗口大小 (如果用户手动指定)
            stride (int, optional): 滑动步长 (如果用户手动指定)
            mode (str): "mean"（均值池化）或 "max"（最大池化）
        """
        super(ChannelSample, self).__init__()
        assert mode in ["mean", "max"], "mode must be 'mean' or 'max'"
        assert out_channels is not None or (kernel_size is not None and stride is not None), \
            "out_channels or (kernel_size, stride) must be specified"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        # if self.out_channels <= self.in_channels:
        #     self.conv = Conv(self.out_channels, self.out_channels, 1, 1)
        # else:
        #     self.conv = Conv(self.out_channels - self.in_channels, self.out_channels - self.in_channels, 1, 1)

    def compute_kernel_stride(self, in_channels, sample_channels):
        """
        如果用户没有手动指定 kernel_size 和 stride，则根据 out_channels 自动计算
        Args:
            in_channels (int): 输入通道数
        Returns:
            kernel_size (int), stride (int)
        """
        if self.kernel_size is not None and self.stride is not None:
            return self.kernel_size, self.stride

        stride = max(1, in_channels // sample_channels)
        kernel_size = in_channels - (sample_channels - 1) * stride

        return kernel_size, stride

    def downsampling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 输入张量，形状为 (N, C, H, W)
        Returns:
            Tensor: 经过通道池化后的张量，形状为 (N, new_C, H, W)
        """
        N, C, H, W = x.shape

        kernel_size, stride = self.compute_kernel_stride(C, self.out_channels)

        # 变换形状使 unfold`作用于 `C` 维度，并保持 `H, W` 形状不变
        x = x.permute(0, 2, 3, 1)  # 变换到 (N, H, W, C)
        x_unfold = x.unfold(dimension=-1, size=kernel_size, step=stride)  # (N, H, W, new_C, kernel_size)

        if self.mode == "mean":
            out = x_unfold.mean(dim=-1)
        elif self.mode == "max":
            out = x_unfold.max(dim=-1)[0]

        # 恢复通道维度到正确的顺序
        out = out.permute(0, 3, 1, 2)  # (N, new_C, H, W)
        # out = self.conv(out)

        return out

    def upsampling(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 输入张量，形状为 (N, C, H, W)
        Returns:
            Tensor: 逐步插入采样后的张量，形状为 (N, C + new_C, H, W)
        """
        N, C, H, W = x.shape

        # 计算 kernel_size 和 stride
        kernel_size, stride = self.compute_kernel_stride(C, self.out_channels - self.in_channels)

        # 计算 new_C
        new_C = (C - kernel_size) // stride + 1
        if new_C <= 0:
            raise ValueError(f"计算得到 new_C={new_C}，请调整 kernel_size 或 stride 使其有效")

        # 变换形状使 `unfold` 作用于 `C` 维度，并保持 `H, W` 形状不变
        x = x.permute(0, 2, 3, 1)  # 变换到 (N, H, W, C)
        x_unfold = x.unfold(dimension=-1, size=kernel_size, step=stride)  # (N, H, W, new_C, kernel_size)

        # 计算池化操作（均值池化 or 最大池化）
        if self.mode == "mean":
            pooled = x_unfold.mean(dim=-1)  # 计算均值 (N, H, W, new_C)
        elif self.mode == "max":
            pooled = x_unfold.max(dim=-1)[0]  # 计算最大值 (N, H, W, new_C)

        # 恢复通道维度到正确的顺序 (N, new_C, H, W)
        pooled = pooled.permute(0, 3, 1, 2)  # (N, new_C, H, W)
        # pooled = self.conv(pooled)

        # **创建插值后的通道张量**
        expanded_channels = []
        orig_channels = x.permute(0, 3, 1, 2)  # (N, C, H, W)

        pooled_idx = 0
        i = 0
        while i < C:
            expanded_channels.append(orig_channels[:, i:i + kernel_size, :, :])  # 原始的 kernel 部分
            i += kernel_size
            if pooled_idx < new_C:
                expanded_channels.append(pooled[:, pooled_idx:pooled_idx + 1, :, :])  # 插入池化值
                pooled_idx += 1

        # 拼接插值后的通道
        output = torch.cat(expanded_channels, dim=1)

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.shape
        if self.out_channels <= C:
            out = self.downsampling(x)
        elif self.out_channels > C:
            out = self.upsampling(x)

        return out


if __name__ == "__main__":
    # 生成测试数据
    N, C, H, W = 1, 4, 4, 4  # batch=1, 通道数=4, 高度=4, 宽度=4
    x = torch.randn(N, C, H, W)  # 随机输入
    print("输入 x:\n", x)

    model = ChannelSample(in_channels=C, out_channels=6, mode="mean")
    out = model(x)

    print("\n输出形状:", out.shape)  # 期望 (1, 2, 4, 4)
    print("\n输出结果:\n", out)
