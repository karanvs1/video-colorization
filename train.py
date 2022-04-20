# Training Class
from model import *
from utils import *
from dataloader import *

import torch.optim as optim
from tqdm import tqdm
import argparse

class VCNetSetup:
    def __init__(self, args):
        self.epochs = args.num_epochs
        self.lr = args.learning_rate
        self.batch_size = args.batch_size
        self.device = args.device

    def _training_loop(self): 
        # Dataloader loop (batches)
        train_items = #Call function from data loader 
        train_loader = torch.utils.data.DataLoader(train_items, batch_size=self.batch_size, shuffle=True, num_workers=4)
        batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train') 
        
        for i, (images, ground_truth) in enumerate(train_loader):
            self.optimizer.zero_grad()

            if(self.device=='gpu'):
                x = x.cuda()
                y = y.cuda()

            output = self.model(images)
            loss = self.criterion(output, ground_truth)
            loss.backward()

            self.optimizer.step()
            total_loss += float(loss)

            # monitor training
            batch_bar.set_postfix( loss="{:.04f}".format(float(total_loss / (i + 1))),
                lr="{:.15f}".format(float(self.optimizer.param_groups[0]['lr'])))

            #save model
            self.save()

    def train(self):  
        # Main training loop (epochs)
        epochs = self.epochs
        device = self.device
    
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            self._training_loop()


    def prepare(self):
        # Model definition, loss, optimizer, scheduler, etc
        self.model = VCNet()
        self.criterion = nn.CrossEntropyLoss(reduce=False) #TODO: Change? 
        self.optimizer = optim.SGD(self.model.parameters(), lr= self.lr)

        pass

    def save(self):
        # Save model
        torch.save(self.model.state_dict(), os.path.join(args.model_path))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument('--num_epochs', type = int, default = 200)
    parser.add_argument('--batch_size', type = int, default = 60)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--learning_rate', type = float, default = 1e-3)

    parser.add_argument('--model_path', type = str, default = 'saved_models/', help = 'path for saving trained models')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'cpu/gpu')

    args = parser.parse_args()

    #Sequential execution:
    trainer = VCNetSetup(args)
    trainer.prepare()
    trainer.train()

    print("Training completed")


        