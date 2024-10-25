import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class HieraSegV2(nn.Module):
    def __init__(self):
        super(HieraSegV2, self).__init__()
        
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_layers = list(resnet.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        encoder_dims = [256, 512, 1024, 2048]
        decoder_dims = [256, 128, 64]
        
        self.decoder3 = nn.ModuleList([
            DecoderBlock(encoder_dims[3], decoder_dims[0]),
            DecoderBlock(decoder_dims[0] + encoder_dims[2], decoder_dims[0]),
            DecoderBlock(decoder_dims[0] + encoder_dims[1], decoder_dims[1]),
            DecoderBlock(decoder_dims[1] + encoder_dims[0], decoder_dims[2])
        ])
        
        self.decoder2 = nn.ModuleList([
            DecoderBlock(encoder_dims[3], decoder_dims[0]),
            DecoderBlock(decoder_dims[0] + encoder_dims[2], decoder_dims[1]),
            DecoderBlock(decoder_dims[1] + encoder_dims[1], decoder_dims[2])
        ])
        
        self.decoder1 = nn.ModuleList([
            DecoderBlock(encoder_dims[3], decoder_dims[0]),
            DecoderBlock(decoder_dims[0] + encoder_dims[2], decoder_dims[1])
        ])
        
        # Final classification heads
        self.final_conv3 = nn.Conv2d(decoder_dims[2], 7, kernel_size=1)  # 7 parts
        self.final_conv2 = nn.Conv2d(decoder_dims[2], 3, kernel_size=1)  # 3 classes
        self.final_conv1 = nn.Conv2d(decoder_dims[1], 2, kernel_size=1)  # 2 classes
        
    def forward(self, x):
        input_shape = x.shape[-2:]  # BxCxHxW
        
        # Store encoder features
        features = []
        for module in self.encoder_layers:
            x = module(x)
            features.append(x)
        # feature embeds from resnest blocks
        e1, e2, e3, e4 = features[4], features[5], features[6], features[7]
        
        # Level 3 decoding (detailed parts)
        x3 = self.decoder3[0](e4)
        x3 = F.interpolate(x3, size=e3.shape[2:], mode='bilinear', align_corners=True)
        # skipcons
        x3 = torch.cat([x3, e3], dim=1)
        x3 = self.decoder3[1](x3)
        x3 = F.interpolate(x3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat([x3, e2], dim=1)
        x3 = self.decoder3[2](x3)
        x3 = F.interpolate(x3, size=e1.shape[2:], mode='bilinear', align_corners=True)
        x3 = torch.cat([x3, e1], dim=1)
        x3 = self.decoder3[3](x3)
        out3 = self.final_conv3(x3)
        
        # Level 2 decoding (upper/lower body) 
        x2 = self.decoder2[0](e4)
        x2 = F.interpolate(x2, size=e3.shape[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat([x2, e3], dim=1)
        x2 = self.decoder2[1](x2)
        x2 = F.interpolate(x2, size=e2.shape[2:], mode='bilinear', align_corners=True)
        x2 = torch.cat([x2, e2], dim=1)
        x2 = self.decoder2[2](x2)
        out2 = self.final_conv2(x2)
        
        x1 = self.decoder1[0](e4)
        x1 = F.interpolate(x1, size=e3.shape[2:], mode='bilinear', align_corners=True)
        x1 = torch.cat([x1, e3], dim=1)
        x1 = self.decoder1[1](x1)
        out1 = self.final_conv1(x1)
        
        # Upscale all outputs to match input resolution
        out3 = F.interpolate(out3, size=input_shape, mode='bilinear', align_corners=True)
        out2 = F.interpolate(out2, size=input_shape, mode='bilinear', align_corners=True)
        out1 = F.interpolate(out1, size=input_shape, mode='bilinear', align_corners=True)
        
        return out1, out2, out3