pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -U -r requirements.txt
pip install -U jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html


# sudo apt-get install fuse gcsfuse -y
# mkdir -p $HOME/gcs-heegyu-kogpt
# gcsfuse heegyu-kogpt $HOME/gcs-heegyu-kogpt