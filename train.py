# Training Class
from model import *
from utils import *
from dataloader2 import VCSamples
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class VCNetSetup:
    def __init__(self, config):
        self.main_config = config
        self.config = config["Setup"]
        self.optimizer_params = self.config["optimizer_params"]
        self.device = self.config["device"]
        self.model_path = os.path.join(config["model_dir"], config["model_name"])
        self._save_params(config["experiment"], config)

    def _save_params(self, experiment_name, metadata):
        try:
            os.mkdir(self.model_path)
        except FileExistsError:
            # d = input("Model name already exists. Delete existing model? (y/n) ")
            d = "y"
            if d == "y":
                import shutil

                shutil.rmtree(self.model_path)
                os.mkdir(self.model_path)
            else:
                print("Error! Exiting!")
                exit(0)

        self.checkpoint_path = os.path.join(self.model_path, "Checkpoints")
        os.mkdir(self.checkpoint_path)
        print("Model to be saved at: ", self.model_path)

        # Saving Model Configuration
        with open(os.path.join(self.model_path, "configuration.yaml"), "w") as metadata:
            yaml.dump({"Experiment": experiment_name}, metadata, indent=4, default_flow_style=False)
            yaml.dump(self.main_config, metadata, indent=4, default_flow_style=False)
        print("Model parameters and configuration saved!")

    def _save_checkpoint(self, epoch, model, optimizer, loss):
        print("Saving Checkpoint!")
        checkpoint_path = os.path.join(self.checkpoint_path, "chkpt_" + str(epoch) + ".pth")
        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "loss": loss},
            checkpoint_path,
        )

    def _training_loop(self, train_loader):
        # Dataloader loop (batches)
        total_loss = 0
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc="Train")

        for i, (images, gt) in enumerate(train_loader):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            images = images.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)

            output = self.model(images)
            # output = np.transpose(output,(0,3,2,1))
            # output = np.transpose(output,(0,1,3,2))
            # print("Output shape: ",output.shape)
            # plt.imshow(images[0][3].detach().numpy())
            # plt.show()
            # plt.imshow(output[0][0].detach().numpy())
            # plt.show()
            # plt.imshow(output[0][1].detach().numpy())
            # plt.show()
            # print("gt shape: ",gt.shape)
            loss = self.criterion(output, gt)
            loss.backward()

            self.optimizer.step()
            total_loss += float(loss.item())

            # monitor training
            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (i + 1))), lr="{:.15f}".format(float(self.optimizer.param_groups[0]["lr"]))
            )
            batch_bar.update()  # Update tqdm bar
            print("something happened")

        batch_bar.close()
        return total_loss

    def train(self):
        # Main training loop (epochs)
        train_dataset = VCSamples(self.config["context"])
        train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=self.config["num_workers"])

        for epoch in range(self.config["epochs"]):
            train_loss = self._training_loop(train_loader)
            self._save_checkpoint(epoch, self.model, self.optimizer, train_loss)
        print("\nTraining Complete")

    def prepare(self):
        torch.backends.cudnn.benchmark = True

        # * Model
        self.model = VCNet(self.main_config).to(self.device)

        # * Loss
        # self.criterion = nn.CrossEntropyLoss(reduce=False) #TODO: Change?
        self.criterion = nn.MSELoss()

        # * Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
        self.scheduler = None

    def save(self):
        print("Saving Model!")
        save_path = os.path.join(self.model_path, "saved_model.pth")
        torch.save(self.model.state_dict(), save_path)


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # Model parameters
#     parser.add_argument('--num_epochs', type = int, default = 200)
#     parser.add_argument('--batch_size', type = int, default = 60)
#     parser.add_argument('--num_workers', type = int, default = 4)
#     parser.add_argument('--learning_rate', type = float, default = 1e-3)

#     parser.add_argument('--model_path', type = str, default = 'saved_models/', help = 'path for saving trained models')
#     parser.add_argument('--device', type = str, default = 'cpu', help = 'cpu/gpu')

#     args = parser.parse_args()

#     #Sequential execution:
#     trainer = VCNetSetup(args)
#     trainer.prepare()
#     trainer.train()

#     print("Training completed")
