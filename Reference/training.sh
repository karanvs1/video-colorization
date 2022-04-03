# Script for training run

# Set up the environment
pip install wheel setup-tools
pip install cupy scikit-image opencv-python

# Mount Google Drive
sudo add-apt-repository ppa:alessandro-strada/ppa
sudo apt update && sudo apt install google-drive-ocamlfuse
google-drive-ocamlfuse

!sudo apt-get install w3m # to act as web browser 
!xdg-settings set default-web-browser w3m.desktop # to set default browser 

! mkdir cmudrive
! google-drive-ocamlfuse cmudrive
! pip install wandb torch-summary
! wandb login 4bdbe9c204105e1264fe3f54df2732fd1fff8040  #chinmay-wandb

# Download the data and weights
!wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip -p /pretrained/DAVIS-2017-trainval-480p.zip   #DAVIS
#TODO command to copy ResNet weights from google drive into pretrained folder
#TODO command to copy VGG16 weights from google drive into pretrained folder