from CurrentNetwork import CurrentNetwork as Network
from params import Params as Params
import os
import time

def create_train_directories(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_dir + 'networks')
    os.mkdir(save_dir + 'results')


def create_test_directories(save_dir):
    os.mkdir(save_dir)
    os.mkdir(save_dir + 'errors')


def train_network(save_dir, hyperParams):
    create_train_directories(save_dir)
    network = Network(save_dir)
    for i in range(0, hyperParams.epochs):
        network.run()


def test_network(save_dir, hyperParams):
    network = Network(save_dir, load_path='/media/rdstorage1/Userdata/Thomasvh/Saved runs/VGG_06070220062016/networks/VGG_epoch60.npz')
    network.test_network()


if __name__ == "__main__":
    hyperParams = Params()
    date_and_time = time.strftime("%S%M%I%d%m%Y")
    save_dir = '/media/rdstorage1/Userdata/Thomasvh/Runs/' + hyperParams.network + '_' + date_and_time + '/'
    if hyperParams.mode == 0:
        train_network(save_dir, hyperParams)
    elif hyperParams == 1:
        test_network(save_dir, hyperParams)


