import torch
import torch.nn as nn
from torchvision.models.segmentation import swin_transformer
from torchvision.ops.misc import Conv2dNormActivation

class HieraSegModel(nn.Module):
    def __init__(self, num_classes, hierarchy_levels, pretrained=True):
        """
        Args:
            num_classes: The number of segmentation classes.
            hierarchy_levels: The number of hierarchy levels for HSSN.
            pretrained: If True, load a pretrained Swin backbone.
        """
        super(HieraSegModel, self).__init__()
        
        # Load the pretrained Swin backbone
        self.backbone = swin_transformer.swin_v2_b(weights="IMAGENET1K_V2" if pretrained else None)
        
        # Modify the number of output channels for the segmentation task
        self.backbone.features[0] = nn.Conv2d(3, 96, kernel_size=(4, 4), stride=(4, 4))  # Example for a change.
        
        # Segmentation Head: Final layer
        self.seg_head = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),  # Reduce features to 512 channels
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_classes * hierarchy_levels, kernel_size=1)  # Output for hierarchical segmentation
        )
        
        # Softmax layer applied over the hierarchy
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # Pass input through the Swin transformer backbone
        x = self.backbone.features(x)
        
        # Pass the output through the segmentation head
        x = self.seg_head(x)
        
        # Apply softmax for hierarchical classification
        x = self.softmax(x)
        
        return x

# Example usage
if __name__ == "__main__":
    model = HieraSegModel(num_classes=21, hierarchy_levels=3, pretrained=True)
    input_tensor = torch.rand(1, 3, 512, 512)  # Example input tensor (Batch_size, Channels, Height, Width)
    output = model(input_tensor)
    print(f"Output shape: {output.shape}")
