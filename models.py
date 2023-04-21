from torch import nn
from blocks import *


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()

        """ Encoder """
        self.e1 = UNetEncoder(in_c, 64)
        self.e2 = UNetEncoder(64, 128)
        self.e3 = UNetEncoder(128, 256)
        self.e4 = UNetEncoder(256, 512)

        """ Bottleneck """
        self.b = Conv(512, 1024)

        """ UNetDecoder """
        self.d1 = UNetDecoder(1024, 512)
        self.d2 = UNetDecoder(512, 256)
        self.d3 = UNetDecoder(256, 128)
        self.d4 = UNetDecoder(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ UNetDecoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs


class SegNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()

        """ Encoder """
        self.e1 = SegNetEncoder(in_c, 64)
        self.e2 = SegNetEncoder(64, 128)
        self.e3 = SegNetEncoder(128, 256)
        self.e4 = SegNetEncoder(256, 512)

        """ Bottleneck """
        self.b = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        """ Decoder """
        self.d4 = SegNetDecoder(1024, 512)
        self.d3 = SegNetDecoder(512, 256)
        self.d2 = SegNetDecoder(256, 128)
        self.d1 = SegNetDecoder(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, out_c, kernel_size=1, padding=0)

    def forward(self, inputs):
        """Encoder"""
        s1, indices1 = self.e1(inputs)
        s2, indices2 = self.e2(s1)
        s3, indices3 = self.e3(s2)
        s4, indices4 = self.e4(s3)

        """ Bottleneck """
        b = self.b(s4)

        """ Decoder """
        d4 = self.d4(b, indices4)
        d3 = self.d3(d4, indices3)
        d2 = self.d2(d3, indices2)
        d1 = self.d1(d2, indices1)

        outputs = self.outputs(d1)

        return outputs


class SemanticUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()

        """ Encoder """
        self.e1 = Conv(in_c, 64)
        self.e2 = Conv(64, 128)
        self.e3 = Conv(128, 256)
        self.e4 = Conv(256, 512)

        """ Bottleneck """
        self.b = Conv(512, 1024)

        """ Decoder """
        self.d1 = UpConv(1024, 512)
        self.d2 = UpConv(512, 256)
        self.d3 = UpConv(256, 128)
        self.d4 = UpConv(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        """Encoder"""
        s1 = self.e1(x)
        s2 = self.e2(nn.functional.max_pool2d(s1, kernel_size=2, stride=2))
        s3 = self.e3(nn.functional.max_pool2d(s2, kernel_size=2, stride=2))
        s4 = self.e4(nn.functional.max_pool2d(s3, kernel_size=2, stride=2))

        """ Bottleneck """
        b = self.b(nn.functional.max_pool2d(s4, kernel_size=2, stride=2))

        """ Decoder """
        d1 = self.d1(b)
        d2 = self.d2(cat([d1, s4], dim=1))
        d3 = self.d3(cat([d2, s3], dim=1))
        d4 = self.d4(cat([d3, s2], dim=1))

        outputs = self.outputs(cat([d4, s1], dim=1))
        return outputs


class AttentionUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()

        """ Encoder """
        self.e1 = Conv(in_c, 64)
        self.e2 = Conv(64, 128)
        self.e3 = Conv(128, 256)
        self.e4 = Conv(256, 512)

        """ Bottleneck """
        self.b = Conv(512, 1024)

        """ Decoder """
        self.d1 = UpConv(1024, 512)
        self.a1 = AttentionBlock(512, 256)
        self.d2 = UpConv(512, 256)
        self.a2 = AttentionBlock(256, 128)
        self.d3 = UpConv(256, 128)
        self.a3 = AttentionBlock(128, 64)
        self.d4 = UpConv(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, out_c, kernel_size=1, padding=0)

    def forward(self, x):
        """Encoder"""
        s1 = self.e1(x)
        s2 = self.e2(nn.functional.max_pool2d(s1, kernel_size=2, stride=2))
        s3 = self.e3(nn.functional.max_pool2d(s2, kernel_size=2, stride=2))
        s4 = self.e4(nn.functional.max_pool2d(s3, kernel_size=2, stride=2))

        """ Bottleneck """
        b = self.b(nn.functional.max_pool2d(s4, kernel_size=2, stride=2))

        """ Decoder """
        d1 = self.d1(b)
        a1 = self.a1(d1, s4)
        d2 = self.d2(a1)
        a2 = self.a2(d2, s3)
        d3 = self.d3(a2)
        a3 = self.a3(d3, s2)
        d4 = self.d4(a3)

        outputs = self.outputs(d4)
        return outputs
