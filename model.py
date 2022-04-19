# MODEL Architecture and declaration
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchsummary import summary


class PreprocessNetwork(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        context = config.context["context"]

    def forward(self, x):
        pass


l_cent = 50.0
l_norm = 100.0
ab_norm = 110.0


def normalize_l(in_l):
    return (in_l - l_cent) / l_norm


def unnormalize_l(in_l):
    return in_l * l_norm + l_cent


def normalize_ab(in_ab):
    return in_ab / ab_norm


def unnormalize_ab(in_ab):
    return in_ab * ab_norm


class Encoder(nn.Module):
    """_summary_"""

    def __init__(self):
        super(Encoder, self).__init__()
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [
            nn.BatchNorm2d(64),
        ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [
            nn.BatchNorm2d(128),
        ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [
            nn.BatchNorm2d(256),
        ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [
            nn.BatchNorm2d(512),
        ]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [
            nn.BatchNorm2d(512),
        ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)

    def forward(self, x):
        x = normalize_l(x)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        return x

    def init_weights(self):
        pass


class Decoder(nn.Module):
    """_summary_"""

    def __init__(self):
        super(Decoder, self).__init__()
        model6 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model6 += [nn.ReLU(True)]
        model6 += [
            nn.BatchNorm2d(512),
        ]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [
            nn.BatchNorm2d(512),
        ]

        model8 = [nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]
        model8 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model8 += [nn.ReLU(True)]

        model8 += [nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0, bias=True)]

        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        self.softmax = nn.Softmax(dim=1)
        self.model_out = nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)
        self.upsample4 = nn.Upsample(scale_factor=4, mode="bilinear")

    def forward(self, x):
        x = self.model6(x)
        x = self.model7(x)
        x = self.model8(x)
        x = self.softmax(x)
        outlab = self.model_out(x)
        outlab = self.upsample4(outlab)
        outlab = unnormalize_ab(outlab)

        return outlab

    def init_weights(self):
        pass


class VCLSTM(nn.Module):
    """_summary_"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def init_weights(self):
        pass


class VCNet(nn.Module):
    """_summary_"""

    def __init__(self):
        super(VCNet, self).__init__()
        # self.preprocess = PreprocessNetwork()
        self.encoder = Encoder()
        self.encoder = load_colorization_weights(model=self.encoder)
        # self.cnnlstm = VCLSTM()
        self.decoder = Decoder()
        self.decoder = load_colorization_weights(model=self.decoder)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def load_colorization_weights(model):
    model_dict = model.state_dict()
    try:
        pretrained_dict = torch.load(r"colorization_weights.pth")
        print(len(list(pretrained_dict.keys())))
    except:
        import torch.utils.model_zoo as model_zoo

        model.load_state_dict(
            model_zoo.load_url(
                "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth",
                map_location="cpu",
                check_hash=True,
                model_dir=r".",
                file_name="colorization_weights.pth",
            )
        )
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print("\nLoaded pretrained weights:", pretrained_dict.keys())
    # print(len(list(pretrained_dict.keys())))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


# if __name__ == "__main__":
#     model = VCNet()
#     print(summary(model, (1, 256, 256)))
