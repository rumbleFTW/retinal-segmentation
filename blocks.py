from torch import nn, cat


class Conv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UpConv(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=2, stride=2):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_c, out_c, kernel_size, stride)

    def forward(self, x):
        x = self.upconv(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = Conv(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p


class SegNetEncoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x, p = self.pool(x)
        return x, p


class UNetDecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = Conv(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class SegNetDecoder(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c // 2, kernel_size=3, padding=1)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(in_c // 2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_c // 2, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, inputs, indices):
        x = self.conv1(inputs)
        x = self.unpool(x, indices)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0, bias=True)
        self.conv3 = nn.Conv2d(out_c, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = self.conv2(x2)
        f = nn.functional.relu(x1 + x, inplace=True)
        f = self.conv3(f)
        f = nn.functional.sigmoid(f)
        return f * x2
