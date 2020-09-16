import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, _C):
        super(Discriminator, self).__init__()
        
        self.NC  = _C.MODEL.NUM_C
        self.NDF = _C.MODEL.NUM_D_FEAT
        
        def block(in_feat, out_feat, kernel_size=4, stride=2, padding=1):
            layers=nn.Sequential(
                nn.Conv2d (in_feat, out_feat, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_feat), #BN with axis="channel index"
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            )
            return layers
        
        self.model = nn.Sequential(
            # input (self.NC) x 64 x 64
            nn.Conv2d(self.NC, self.NDF, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),    # (NDF*1) x 32 x 32
            *block(self.NDF * 1, self.NDF * 2), # (NDF*2) x 16 x 16
            *block(self.NDF * 2, self.NDF * 4), # (NDF*4) x 8 x 8
            *block(self.NDF * 4, self.NDF * 8), # (NDF*8) x 4 x 4
            nn.Conv2d(self.NDF * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),                       # output (NDF*8) * 1 * 1
        )

    def forward(self, img):
        return self.model(img)