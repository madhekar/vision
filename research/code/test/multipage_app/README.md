sudo docker build -t zmedia .
sudo docker run -p 8501:8501 zmedia

sudo apt-get clean autoclean
sudo apt-get autoremove --yes
rm -rf /var/lib{apt,dpkg,cache,log}
