# MODEL Architecture and declaration
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn

from torchsummary import summary
from utils import *


class PreprocessNetwork(nn.Module):
    def __init__(self, config):
        super(PreprocessNetwork, self).__init__()
        self.config = config
        self.context = self.config["context"]

        sim_layers = []
        for _ in range(self.context * 2):
            sim_layers.append(
                nn.Conv2d(in_channels=2, out_channels=1, kernel_size=self.config['sim_kernel_size'], stride=1, padding='same', bias=True),
                )
        self.similarity = nn.ModuleList(sim_layers)
        self.activation = nn.Sigmoid()
        self.mixing = nn.Conv2d(in_channels=2 * self.context + 1, out_channels=1, kernel_size=self.config['mix_kernel_size'],stride=1, padding='same', bias=True)

        self.initialize_weights()

    def forward(self, x):   # B x 7 x 256 x 256
        base_frame = x[:, self.context].unsqueeze(1) # B x 1 x 256 x 256
        pairwise_processed = []
        idx = 0
        for c in range(self.context * 2 + 1):
            if c != self.context:
                print("Frame : ", c)
                check_frame = x[:, c].unsqueeze(1) # B x 1 x 256 x 256
                if c > self.context:
                    frame_pair = torch.cat((check_frame , base_frame), dim=1)
                else:
                    frame_pair = torch.cat((base_frame, check_frame), dim=1)
                    
                processed_frame = self.activation(self.similarity[idx](frame_pair))

                elementwise_frame = processed_frame * check_frame
                pairwise_processed.append(elementwise_frame)
                idx += 1

        combined_x = torch.cat((
            *pairwise_processed[:self.context],     # B x 3 x 256 x 256
            base_frame,                             # B x 1 x 256 x 256
            *pairwise_processed[self.context:]      # B x 3 x 256 x 256
            ), dim=1)   # B x 7 x 256 x 256

        processed_x = self.mixing(combined_x)
        return processed_x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
               nn.init.constant_(m.weight, 1)
               nn.init.constant_(m.bias, 0)


class Encoder(nn.Module):
    """_summary_"""

    def __init__(self):
        super(Encoder, self).__init__()
        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True)]
        model1 += [nn.ReLU(True)]
        model1 += [
            nn.BatchNorm2d(64)       ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [
            nn.BatchNorm2d(128)
        ]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [
            nn.BatchNorm2d(256)
        ]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [
            nn.BatchNorm2d(512)
        ]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [
            nn.BatchNorm2d(512)
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
            nn.BatchNorm2d(512)        ]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [
            nn.BatchNorm2d(512)        ]

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
    def __init__(self, config):
        super(VCNet, self).__init__()
        self.preprocess = PreprocessNetwork(config["PreprocessNet"])
        self.encoder = Encoder()
        self.encoder = load_colorization_weights(model=self.encoder)
        # self.cnnlstm = VCLSTM()
        self.decoder = Decoder()
        self.decoder = load_colorization_weights(model=self.decoder)

    def forward(self, x):
        x = self.preprocess(x)
        print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)

        return x


def load_colorization_weights(model):
    model_dict = model.state_dict()
    try:
        pretrained_dict = torch.load(r"colorization_weights.pth")
        # print("len dict keys: ", len(list(pretrained_dict.keys())))
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


if __name__ == "__main__":
    with open('test_config.yaml', "r") as f:
        config = yaml.safe_load(f)
    model = VCNet(config)
    print(summary(model, (7, 256, 256)))

 

