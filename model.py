# MODEL Architecture and declaration
from distutils.command.config import config
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
            layer = nn.Sequential(
                nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=True), nn.ReLU(), nn.Conv2d(64, 1, kernel_size=1, bias=True), nn.Sigmoid()
            )
            sim_layers.append(layer)

        self.similarity = nn.ModuleList(sim_layers)
        self.mixing = nn.Conv2d(
            in_channels=2 * self.context + 1,
            out_channels=2 * self.context + 1,
            kernel_size=self.config["mix_kernel_size"],
            stride=1,
            padding="same",
            bias=True,
        )

        self.initialize_weights()

    def forward(self, x, mode):  # B x 7 x 256 x 256
        if mode == "context":
            return x
        base_frame = x[:, self.context].unsqueeze(1)  # B x 1 x 256 x 256
        pairwise_processed = []
        idx = 0
        for c in range(self.context * 2 + 1):
            if c != self.context:
                check_frame = x[:, c].unsqueeze(1)  # B x 1 x 256 x 256
                if c > self.context:
                    frame_pair = torch.cat((check_frame, base_frame), dim=1)
                else:
                    frame_pair = torch.cat((base_frame, check_frame), dim=1)

                elementwise_frame = self.similarity[idx](frame_pair) * check_frame
                pairwise_processed.append(elementwise_frame)
                idx += 1

        combined_x = torch.cat(
            (*pairwise_processed[: self.context], base_frame, *pairwise_processed[self.context :]), dim=1
        )  # B x 7 x 256 x 256

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

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.context = config["context"]
        self.preprocess = PreprocessNetwork(config)

        pre_model1 = [
            nn.Conv2d((self.context * 2 + 1) * 64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        ]

        model1 = [
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
        ]

        model2 = [nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=True)]
        model2 += [nn.ReLU(True)]
        model2 += [nn.BatchNorm2d(128)]

        model3 = [nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=True)]
        model3 += [nn.ReLU(True)]
        model3 += [nn.BatchNorm2d(256)]

        model4 = [nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model4 += [nn.ReLU(True)]
        model4 += [nn.BatchNorm2d(512)]

        model5 = [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=True)]
        model5 += [nn.ReLU(True)]
        model5 += [nn.BatchNorm2d(512)]

        self.pre_model1 = nn.Sequential(*pre_model1)
        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)

    def forward(self, x, mode):
        # print(header(mode))
        x = normalize_l(x)
        if mode == "cic":
            # print(x.shape)
            assert x.size(1) == 1, "Should be a single grayscale frame"
            x = self.model1(x)
        elif mode == "context" or mode == "attention":

            x = x.squeeze(0)
            assert x.size(1) > 1, "Should be a set of grayscale frames"
            x = self.preprocess(x, mode)

            model1_out = torch.empty((x.shape[0], x.shape[1] * 64, 128, 128))
            for i in range(x.shape[1]):
                # print("x channel ", x[:, i].unsqueeze(1).shape)
                l1_out = self.model1(x[:, i, :, :].unsqueeze(1))
                # print("l1 out ", l1_out.shape)
                torch.cat((model1_out, l1_out), axis=1)
                # print("model 1 ", model1_out.shape)

            x = self.pre_model1(model1_out)

        x = self.model2(x)
        x = self.model3(x)
        x = self.model4(x)
        x = self.model5(x)
        return x


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
        model6 += [nn.BatchNorm2d(512)]

        model7 = [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)]
        model7 += [nn.ReLU(True)]
        model7 += [nn.BatchNorm2d(512)]

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
        probs = self.softmax(x)
        # outlab = x
        outlab = self.model_out(probs)
        outlab = self.upsample4(outlab)
        outlab = unnormalize_ab(outlab)

        return outlab, probs


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

    def __init__(self, config, run_mode="train"):
        self.mode = config["Setup"]["model_mode"]
        print("Model MODE: ", self.mode)
        super(VCNet, self).__init__()
        self.encoder = Encoder(config["Encoder"])
        if run_mode == "train":
            self.encoder = load_colorization_weights(model=self.encoder)
        # self.cnnlstm = VCLSTM()
        self.decoder = Decoder()
        if run_mode == "train":
            self.decoder = load_colorization_weights(model=self.decoder)

    def forward(self, x, mode="context"):
        x = self.encoder(x, self.mode)
        x = self.decoder(x)

        return x


def load_colorization_weights(model):
    model_dict = model.state_dict()
    try:
        pretrained_dict = torch.load(r"./colorization_weights.pth")

        print("Loaded pretrained weights")
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
        pretrained_dict = torch.load(r"./colorization_weights.pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # print("\nLoaded pretrained weights:", pretrained_dict.keys())
    # print(len(list(pretrained_dict.keys())))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


if __name__ == "__main__":
    with open("attention_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # for k, v in config.items():
    #     print(f"{k}: {v}")
    model = VCNet(config).to(config["Setup"]["device"])

    for param in model.parameters():
        param.requires_grad = False

    for param in model.encoder.preprocess.parameters():
        param.requires_grad = True

    for param in model.encoder.model1.parameters():
        param.requires_grad = True

    for param in model.encoder.pre_model1.parameters():
        param.requires_grad = True

    # summary(model, torch.zeros((2, 9, 256, 256)).cuda())

    summary(model, (7, 256, 256))
