# Training Class
import os
import wandb
import matplotlib.pyplot as plt
import torch.optim as optim
import datetime
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda import amp

from model import *
from utils import *
from dataloader import VCSamples


class VCNetSetup:
    def __init__(self, config):
        self.main_config = config
        self.config = config["Setup"]
        self.optimizer_params = self.config["optimizer_params"]
        self.device = self.config["device"]
        self.model_path = os.path.join(config["model_dir"], config["model_name"])
        self._save_params(config["experiment"], config)

        try:
            os.mkdir("saved_models")
        except:
            print("saved_models already exists!")

    def _save_params(self, experiment_name, metadata):
        try:
            os.mkdir(self.model_path)
        except FileExistsError:
            d = input("Model name already exists. Delete existing model? (y/n) ")
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
        torch.cuda.empty_cache()
        for i, (images, gt) in enumerate(train_loader):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            images = images.to(self.device, non_blocking=True)
            gt = gt.to(self.device, non_blocking=True)

            # with amp.autocast():
            output, _ = self.model(images, self.config["model_mode"])

            # loss = self.criterion(output, gt.requires_grad_(True))
            loss = self.criterion(output, gt)

            loss.backward()
            self.optimizer.step()
            # self.scaler.scale(loss).backward()
            # self.scaler.step(self.optimizer)
            # self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += float(loss.item())
            batch_bar.set_postfix(
                loss="{:.04f}".format(float(total_loss / (i + 1))), lr="{:.04f}".format(float(self.optimizer.param_groups[0]["lr"]))
            )
            batch_bar.update()
        batch_bar.close()

        return total_loss

    def train(self):
        epochs = self.config["epochs"]
        delta_time = datetime.timedelta(seconds=0)

        if self.config["wandb_log"]:
            wandb.init(project="test-project", entity="acvc", config=self.main_config)
            wandb.watch(self.model, criterion=self.criterion, log="all", log_freq=self.config["batch_size"], idx=None)

        for epoch in range(epochs):
            start_time = time.time()
            print("\n" + "-" * 40 + " Epoch " + str(epoch + 1) + " " + "-" * 40)
            train_loss = self._training_loop(self.train_loader)
            print(
                "Epoch {}/{}: Loss {:.04f}, Learning Rate {:.04f}".format(
                    epoch + 1, epochs, float(train_loss / len(self.train_loader)), float(self.optimizer.param_groups[0]["lr"])
                )
            )
            if self.config["wandb_log"]:
                wandb.log(
                    {"Train Loss": float(train_loss / len(self.train_loader)), "Train Learning Rate": self.optimizer.param_groups[0]["lr"]}
                )

            # self._save_checkpoint(epoch, self.model, self.optimizer, train_loss)
            self.save(epoch)
            delta_time += datetime.timedelta(seconds=(time.time() - start_time))
            print(f"Time lapsed = {str(delta_time)}")
            print(f"Time left = {str(delta_time * (epochs - epoch - 1) / (epoch + 1))}")

        print("\nTraining Complete!")

    def prepare(self):
        torch.backends.cudnn.benchmark = True

        # * Dataloader
        train_dataset = VCSamples(self.main_config)
        self.train_loader = DataLoader(train_dataset, batch_size=self.config["batch_size"], **self.config["dataloader_params"])

        # * Model
        self.model = VCNet(self.main_config, run_mode="train").to(self.device)
        self.scaler = amp.GradScaler()

        print("Freezing CIC Weights")

        # * Model Layers (Freeze)
        for param in self.model.parameters():
            param.requires_grad = False

        # * CIC Encoder Layer (Unfreeze)
        for param in self.model.encoder.model1.parameters():
            param.requires_grad = True

        # * Context and Attention Encoder Layer (Unfreeze)
        for param in self.model.encoder.preprocess.parameters():
            param.requires_grad = True
        for param in self.model.encoder.pre_model1.parameters():
            param.requires_grad = True
        summary(self.model, (self.config["context"] * 2 + 1, 256, 256))

        # * Loss
        # self.criterion = nn.CrossEntropyLoss(reduce=False) #TODO: Change?
        self.criterion = nn.MSELoss()

        # * Optimizer
        if self.config["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
        elif self.config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.optimizer.param_groups[0]["lr"], **self.optimizer_params)
        elif self.config["optimizer"] == "RMS":
            self.optimizer = optim.RMSprop(self.model.parameters(), **self.optimizer_params)

        # * Scheduler
        if self.config["scheduler"] == "MultiStepLR":
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, **self.config["sched_params"])
        elif self.config["scheduler"] == "CALR":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config["epochs"] * len(self.train_loader))
            # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **self.config["sched_params"])
        else:
            self.scheduler = None

        if self.config["wandb_log"]:
            wandb.finish()

    def save(self, epoch=None):
        print("Saving Model!")
        if epoch is not None and epoch % self.config["save_freq"] == 0:
            save_path = os.path.join(self.model_path, "saved_model_" + str(epoch + 1) + ".pth")
            torch.save(self.model.state_dict(), save_path)
        else:
            save_path = os.path.join(self.model_path, "saved_model.pth")
            torch.save(self.model.state_dict(), save_path)
