sudo add-apt-repository ppa:alessandro-strada/ppa
sudo apt update && sudo apt install google-drive-ocamlfuse
google-drive-ocamlfuse
sudo apt-get install w3m # to act as web browser 
xdg-settings set default-web-browser w3m.desktop # to set default browser 

cd deeplearning
mkdir cmudrive
cd ..
google-drive-ocamlfuse /deeplearning/cmudrive