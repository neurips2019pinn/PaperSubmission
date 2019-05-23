## Pose Extraction Networks

This is a the implementation of Pose Extraction Networks

#### Install
To copy and setup the project
```
git clone https://github.com/rsimmons1/WhiteningNN.git
# cd into newly cloned directory
mkdir data
sudo python3 setup.py install
```

To install the AffNIST, navigate to the `./data` folder and run the following (takes a bit)
```
wget https://www.cs.toronto.edu/~tijmen/affNIST/32x/transformed/training_and_validation_batches.zip
unzip -a training_and_validation_batches.zip
rm training_and_validation_batches.zip
```

To install smallNORB, navigate to directory './data'
```
mkdir small_norb_root
cd small_norb_root
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz
wget https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz
gunzip *.gz
rm *.gz
```

#### Training

For training on AffNIST dataset run
```
python3 train.py --dataset AffNIST --display
```
