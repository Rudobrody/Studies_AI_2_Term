import torch
import torch.nn as nn


# Class inherits from torch.nn.Module
class ResidualBlock(nn.Module):
    """
    A single residual block for the Generator.
    
    Args:
        in_features (int): The number of input channels
    """

    def __init__(self, in_features: int):
        # "Super() function let me call methods from a parent or base class without explicitly naming it 
        # The line below is calling the constructor of the parent class - nn.Module
        super(ResidualBlock, self).__init__() # Equally super().__init__()
        """
        Notes:
            Sequentail block defines two convolutional layers
            nn.Conv2d: Applies a 2D convolution - learns filters. The same number of input and output esnures shape compatibility for residual addition later
            nn.BatchNorm2d: Normalizes features maps
            nn.PReLU: Parametric ReLU, more flexible 
            Padding=1 ensures the same size of the input and output
        """
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
        )

    def forward(self, x):
        """
        A function for making forward step. The input x goes through convolutional block - the learned trasnformation. Then the oroginal input x is added to
        the output of the conv block"

        Args:
            x: input data
        Returns:
            Output value of the convolutional block 
        """
        return x + self.conv_block(x)
    

class GeneratorResNet(nn.Module):
    """
    Generator network based on residual blocks.

    Args:
        in_channels (int): Number of input image channels.
        out_channels (int): Number of output image channels.
        n_residual_blocks (int): Number of residual blocks.
        upscale_factor (int): The supper-resolution upscale factor (takes a value: {2, 4})
    
    Notes:
        Why don't we normalize values in the first layer? because normalization in the first layer can destroy raw input statistics.
        In image tasks, the first convolutional learn low-level features like edges, textures and colors. The same situation happens in the last layer.
        We use normalization in second layer, because second layer is working on features maps. 
        Each block of the ResiudalBlock refines feature 
        Second Conv layer merges all residual outputs into a single feature representation
        Each block in upsampling increases channgels 64 -> 256
    """
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=8, upscale_factor=4):
        super(GeneratorResNet, self).__init__() # Equally super().__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU()) # Large kernel to capture wide spatial context

        # Residual blocks
        res_blocks = [ResidualBlock(64) for _ in range(n_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64))

        # Upsampling layer (PixelSchuffle)
        upsampling = []
        for _ in range(upscale_factor // 2): # Run twice for 4x
            upsampling += [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            ]       
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())


    def forward(self, x):
        """
        Function for making step forward in the generator

        Args:
            x: input data, output from the ResidualBlock 
        """
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2) # Element-wise sum
        out = self.upsampling(out)
        out = self.conv3(out)
        return out