# MLP-Mixer in Pytorch! :art:

An implementation of **MLP-Mixer** or **Mixer** in short in Pytorch. Mixer is a deep learning architecture for vision that performs comparable with S.O.T.A. CNN-based and attention-based models, while `only using MLP building blocks`. Mixer uses 3 types of MLP throughout its architecture:
1. `Per-patch MLP`: projects patch pixels into a tensor shaped as `[patches x channels]`
2. `Token-mixing MLP`: mixes spatial features per channel
3. `Channel-mixing MLP`: mixes features across channel per spatial location

Note: In the paper, the terms `tokens` and `patches` are used interchangeably.

# Usage
```
python train.py
```
**Options**
* `--patch_size`: size of patches of input image
* `--n_layers`: number of mixer layers in the Mixer Stack 
* `--n_channel`: dimension of channel you want to project the pixels to
* `--n_hidden`: hidden dimension of mlp inside mlp layers
* `--dataset`: choose from `mnist`, `cifar10`, or `cifar100`


# Quick Implementation

## MLP-Mixer Architecture Overview
The whole MLP-Mixer expects a input tensor of shape `[B,C,H,W]` where `B` is the batch size, `C` is the number of channels of the input image, and `H` and `W` are the height and width. `ImageToPatch` divides the input image tensors into patches of size `P`. `PerPatchMLP` takes as input patches of pixels and projects them to channel dimension resulting to tensor of shape `[B, n_tokens, n_channel]`. `MixerStack` is composed of `N` layers of `MixerLayers` composed of `Token-mixing MLP` and `Channel-mixing MLP` for mixing the feautures along the token and channel dimension respectively. 
```python
class MLP_Mixer(nn.Module):
    def __init__(self, n_layers, n_channel, n_hidden, n_output, image_size, patch_size, n_image_channel):
        super().__init__()

        n_tokens = (image_size // patch_size)**2
        n_pixels = n_image_channel * patch_size**2

        self.ImageToPatch = ImageToPatches(patch_size = patch_size)
        self.PerPatchMLP = PerPatchMLP(n_pixels, n_channel)
        self.MixerStack = nn.Sequential(*[
            nn.Sequential(
                TokenMixingMLP(n_tokens, n_channel, n_hidden),
                ChannelMixingMLP(n_tokens, n_channel, n_hidden)
            ) for _ in range(n_layers)
        ])
        self.OutputMLP = OutputMLP(n_tokens, n_channel, n_output)

    def forward(self, x):
        x = self.ImageToPatch(x)
        x = self.PerPatchMLP(x)
        x = self.MixerStack(x)
        return self.OutputMLP(x)
```
## Image to Patches and PerPatchMLP

`PerPatchMLP` projects pixels to channel dimension. It takes as input a tensor of shape `[B, n_tokens, n_pixels]` and projects to `[B, n_tokens, n_channel]`.
```python
class ImageToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.P = patch_size

    def forward(self, x):
        P = self.P
        B,C,H,W = x.shape                       # [B,C,H,W]                 
        x = x.reshape(B,C, H//P, P , W//P, P)   # [B,C, H//P, P, W//P, P]  
        x = x.permute(0,2,4, 1,3,5)             # [B, H//P, W//P, C, P, P]  
        x = x.reshape(B, H//P * W//P, C*P*P)    # [B, H//P * W//P, C*P*P]  
                                                # [B, n_tokens, n_pixels]
        return x

class PerPatchMLP(nn.Module):
    def __init__(self, n_pixels, n_channel):
        super().__init__()
        self.mlp = nn.Linear(n_pixels, n_channel)

    def forward(self, x):      
        return self.mlp(x)  # [B, n_tokens, n_channel]    
```

## Token-mixing MLP
`Token-mixing MLP` projects tokens to hidden dimension and back to token dimension. Therefore it expects an input of shape `[B, n_channel, n_tokens]`, which is done by swapping the axes.
```python
class TokenMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.mlp1 = nn.Linear(n_tokens, n_hidden)       
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(n_hidden, n_tokens)

    def forward(self, X):
        z = self.layer_norm(X)                  # z:    [B, n_tokens, n_channel]
        z = z.permute(0, 2,1)                   # z:    [B, n_channel, n_tokens]
        z = self.gelu(self.mlp1(z))             # z:    [B, n_channel, n_hidden] 
        z = self.mlp2(z)                        # z:    [B, n_channel, n_tokens]
        z = z.permute(0, 2,1)                   # z:    [B, n_tokens, n_channel]
        U = X + z                               # U:    [B, n_tokens, n_channel]
        return U
```

## Channel-mixing MLP 
`Channel-mixing MLP` projects channels to hidden dimension and back to channel dimension. Since the input tensor has shape `[B, n_tokens, n_channel]`, there is no need to swap axes. 
```python
class ChannelMixingMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_hidden):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.mlp3 = nn.Linear(n_channel, n_hidden)
        self.gelu = nn.GELU()
        self.mlp4 = nn.Linear(n_hidden, n_channel)

    def forward(self, U):
        z = self.layer_norm(U)                  # z: [B, n_tokens, n_channel]
        z = self.gelu(self.mlp3(z))             # z: [B, n_tokens, n_hidden]
        z = self.mlp4(z)                        # z: [B, n_tokens, n_channel]
        Y = U + z                               # Y: [B, n_tokens, n_channel]
        return Y
```

## OutputMLP
`OutputMLP` is the usual fully-connected for outputs. The only difference is it takes as input features averaged along the token dimension.
```python
class OutputMLP(nn.Module):
    def __init__(self, n_tokens, n_channel, n_output):
        super().__init__()
        self.layer_norm = nn.LayerNorm([n_tokens, n_channel])
        self.out_mlp = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.layer_norm(x)                  # x: [B, n_tokens, n_channel]
        x = x.mean(dim=1)                       # x: [B, n_channel] 
        return self.out_mlp(x)                  # x: [B, n_output]
```

# MLP-Mixer vs CNN
Actually, all MLP building blocks in MLP can be constructed using ConvLayers
* `Per-Patch MLP` -> Conv Layer with `kernel of size PxP` and `stride of size P`
* `Token-mixing MLP` -> Single-channel Conv Layer with `kernel of size of the full receptive field (i.e. H/P x W/P)`
* `Channel-mixing MLP` -> Conv Layer with `kernel of size 1x1`
