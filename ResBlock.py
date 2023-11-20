
class ResBlock(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size):
        super(ResBlock, self).__init__()

        self.function=nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, padding=1),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=1)
        )

        self.downsample=nn.Sequential(
            nn.Conv2d(input_channels,out_channels,kernel_size,padding=1)
        )

    def forward(self, x):
        identify = x
        identify=self.downsample(identify)

        f = self.function(x)
        out = f + identify
        return out
