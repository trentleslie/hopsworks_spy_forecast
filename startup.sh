sudo apt update
sudo apt upgrade
sudo apt install nginx
sudo systemctl start nginx
sudo systemctl enable nginx
sudo ufw allow 'Nginx Full'
sudo apt install python3 python3-pip
sudo apt install git
git config --global user.name "trentleslie"
git config --global user.email "trentleslie@gmail.com"
ssh-keygen -t ed25519 -C "trentleslie@gmail.com"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda create --name hopsworks python=3.9
sudo apt install build-essential
conda activate hopsworks