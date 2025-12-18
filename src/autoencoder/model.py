import torch
import torch.nn as nn
import torch.nn.functional as F

#block dasar: Residual Block, Downsample Block, Upsample Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
    
    def forward(self, x):
        return x + self.seq(x)

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.seq(x)


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.seq(x)

#encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=64, latent_dim=128):
        super(Encoder, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.res_blocks1 = nn.Sequential(
            ResidualBlock(base_filters),
            ResidualBlock(base_filters)
        )

        self.down1 = DownsampleBlock(base_filters, base_filters * 2)

        self.res_blocks2 = nn.Sequential(
            ResidualBlock(base_filters * 2),
            ResidualBlock(base_filters * 2)
        )

        self.down2 = DownsampleBlock(base_filters * 2, latent_dim)

        self.bottleneck = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim)
        )
    
    def forward(self, x):
        x = self.initial(x)

        x = self.res_blocks1(x)
        x = self.down1(x)
        
        x = self.res_blocks2(x)
        x = self.down2(x)

        x = self.bottleneck(x)
        
        return x

#decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=128, base_filters=64, out_channels=3, scale=4):
        super(Decoder, self).__init__()
        
        self.scale = scale

        self.bottleneck = nn.Sequential(
            ResidualBlock(latent_dim),
            ResidualBlock(latent_dim)
        )

        self.up1 = UpsampleBlock(latent_dim, base_filters * 2)

        self.res_blocks1 = nn.Sequential(
            ResidualBlock(base_filters * 2),
            ResidualBlock(base_filters * 2)
        )

        self.up2 = UpsampleBlock(base_filters * 2, base_filters)

        self.res_blocks2 = nn.Sequential(
            ResidualBlock(base_filters),
            ResidualBlock(base_filters)
        )

        upscale_blocks = []
        for _ in range(int(torch.log2(torch.tensor(scale)).item())):
            upscale_blocks.extend([
                UpsampleBlock(base_filters, base_filters),
                ResidualBlock(base_filters)
            ])
        self.upscale = nn.Sequential(*upscale_blocks)

        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.bottleneck(x)

        x = self.up1(x)
        x = self.res_blocks1(x)
        
        x = self.up2(x)
        x = self.res_blocks2(x)

        x = self.upscale(x)

        x = self.final(x)

        x = torch.clamp(x, 0, 1)
        
        return x


class SuperResolutionAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_filters=64, latent_dim=128, scale=4):
        super(SuperResolutionAutoencoder, self).__init__()
        
        self.scale = scale

        self.encoder = Encoder(
            in_channels=in_channels,
            base_filters=base_filters,
            latent_dim=latent_dim
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            base_filters=base_filters,
            out_channels=in_channels,
            scale=scale
        )
    
    def forward(self, x):
        latent = self.encoder(x)

        out = self.decoder(latent)
        
        return out
    
    def get_latent(self, x):
        return self.encoder(x)

def create_model(config):
    model = SuperResolutionAutoencoder(
        in_channels=config.INPUT_CHANNELS,
        base_filters=config.BASE_FILTERS,
        latent_dim=config.LATENT_DIM,
        scale=config.SCALE_FACTOR
    )
    
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total

if __name__ == "__main__":
    print("Testing Autoencoder Model...")

    model = SuperResolutionAutoencoder(
        in_channels=3,
        base_filters=64,
        latent_dim=128,
        scale=4
    )

    batch_size = 2
    lr_size = 64
    hr_size = lr_size * 4
    
    dummy_input = torch.randn(batch_size, 3, lr_size, lr_size)
    
    print(f"\nInput shape: {dummy_input.shape}")

    with torch.no_grad():
        output = model(dummy_input)
        latent = model.get_latent(dummy_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")

    num_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (float32)")

    assert output.shape == (batch_size, 3, hr_size, hr_size), \
        f"Expected output shape {(batch_size, 3, hr_size, hr_size)}, got {output.shape}"
    
    print("\n✓ Model test passed!")
