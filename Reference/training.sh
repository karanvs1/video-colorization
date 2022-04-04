# Script for training run

# Set up the environment
pip install wheel setup-tools
pip install cupy scikit-image opencv-python

# Mount Google Drive
sudo add-apt-repository ppa:alessandro-strada/ppa
sudo apt update && sudo apt install google-drive-ocamlfuse
google-drive-ocamlfuse

sudo apt-get install w3m # to act as web browser 
sudo apt install xdg-utils #The next command doesn't work if this isn't done
xdg-settings set default-web-browser w3m.desktop # to set default browser 

mkdir cmudrive
google-drive-ocamlfuse cmudrive
pip install wandb torch-summary
# wandb login 4bdbe9c204105e1264fe3f54df2732fd1fff8040  #chinmay-wandb

# Download the data and weights:

#DAVIS
mkdir ~/video-colorization/Reference/models
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip   
unzip DAVIS-2017-trainval-480p.zip -d ~/video-colorization/Reference/models/

#ResNet:
mv cmudrive/11785_CMU/project_data/prev_data/resnet50_in_rgb_epoch150_bs512.pth ~/video-colorization/Reference/models/

#VGG16:
wget "https://hkustconnect-my.sharepoint.com/:u:/g/personal/cleiaa_connect_ust_hk/EZeGsvuqh1dJr0E2Fxf6IKoBQ7wZpGi3NFqZxhzC8-3GHg?e=LLWUZT&download=1"
mv EZeGsvuqh1dJr0E2Fxf6IKoBQ7wZpGi3NFqZxhzC8-3GHg\?e\=LLWUZT\&download\=1 VGG_Model.zip
unzip VGG_Model.zip -d ~/video-colorization/Reference/models/