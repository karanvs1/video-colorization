# Testing Class
from model import *
from utils import *
from dataloader import VCSamples_Test
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from tqdm import tqdm

class VideoColorizer:
    def __init__(self, config):
        self.config = config

    def test(self, test_loader):
        # Main testing loop (epochs)
        #make a directory to save the results
        os.makedirs(self.config["output_location"], exist_ok=True)
        for i, img in tqdm(enumerate(test_loader)):
            out, _ = self.model(img)
            # print(img.shape)
            output_image = postprocess_tens(img[:,:,config['Setup']['context'],:,:], out)            # FOR CONTEXT and  ATTENTION 
            # output_image = postprocess_tens(img[:,:,:,:], out)                                     #FOR CIC
            plt.imsave(os.path.join(self.config["output_location"], str(i) + ".png"), output_image)

    def prepare(self):
        # Model definition
        self.model = VCNet(self.config, run_mode = 'test')
        # self.model.load_state_dict(torch.load(r"./saved_models/context3_davis/saved_model.pth")) #don't load for cic
        dictionary = torch.load(r"./saved_models/attention_davis/Checkpoints/chkpt_72.pth")
        self.model.load_state_dict(dictionary["model_state_dict"])
        self.model.eval()

    def save_results(self):
        # Save results
        pass

    def predict(self):
        # Single batch prediction
        pass

if __name__ == "__main__":
    with open("test_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    test_dataset = VCSamples_Test(config)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config["Setup"]["batch_size"], shuffle=False, num_workers=config["Setup"]["num_workers"])
    vc = VideoColorizer(config)
    vc.prepare()
    vc.test(test_loader)
