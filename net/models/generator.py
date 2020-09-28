import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, _C):
        super(Generator, self).__init__()

        self.NZ = _C.MODEL.NUM_Z
        self.NGF = _C.MODEL.NUM_G_FEAT
        self.NC = _C.MODEL.NUM_C

        def block(in_feat, out_feat, kernel_size=4, stride=2, padding=1):
            layers = nn.Sequential(
                nn.ConvTranspose2d(
                    in_feat, out_feat, kernel_size, stride, padding, bias=False
                ),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.2, inplace=True),
            )
            return layers

        self.model = nn.Sequential(
            # input is Z, going into a generator
            *block(
                self.NZ, self.NGF * 8, kernel_size=4, stride=1, padding=0
            ),  # (NGF*8) x  4 x  4
            *block(self.NGF * 8, self.NGF * 4),  # (NGF*4) x  8 x  8
            *block(self.NGF * 4, self.NGF * 2),  # (NGF*2) x 16 x 16
            *block(self.NGF * 2, self.NGF * 1),  # (NGF*1) x 32 x 32
            nn.ConvTranspose2d(self.NGF, self.NC, 4, 2, 1, bias=False),
            nn.Tanh(),  #  (NC) x 32 x 32
        )

    def forward(self, z):
        return self.model(z)
