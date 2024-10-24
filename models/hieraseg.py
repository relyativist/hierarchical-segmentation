import torch.nn as nn
import torchvision.models as models


class HierarchicalSegmentationModel(nn.Module):
    def __init__(self):
        super(HierarchicalSegmentationModel, self).__init__()
        self.encoder = models.resnet50(weights="IMAGENET1K_V1")
        self.encoder_layers = list(self.encoder.children())[:-2]
        self.encoder = nn.Sequential(*self.encoder_layers)
        
        # Decoders for each level
        self.decoder_level1 = nn.Conv2d(2048, 2, kernel_size=1)
        self.decoder_level2 = nn.Conv2d(2048, 3, kernel_size=1)
        self.decoder_level3 = nn.Conv2d(2048, 7, kernel_size=1)
        
        # Upsampling layers
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        features = self.encoder(x)
        
        out_level1 = self.decoder_level1(features)
        out_level2 = self.decoder_level2(features)
        out_level3 = self.decoder_level3(features)
        
        out_level1 = self.upsample(out_level1)
        out_level2 = self.upsample(out_level2)
        out_level3 = self.upsample(out_level3)
        
        return out_level1, out_level2, out_level3
