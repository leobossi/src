
import torch
from utils import get_dataset, average_weights, exp_details
from options import args_parser
import matplotlib.pyplot as plt
from torch import nn

'''
test = True

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork.self).__init__()
        self.flatten = nn.Flatten

'''

if __name__ == '__main__':

    args = args_parser()   #Setting the arguments (from the options.py file)
    exp_details(args)

    deivce = 'cuda'

    train_dataset, test_dataset, user_groups = get_dataset(args)
    print(train_dataset.targets.size())
    print(train_dataset.data.size())


    plt.imshow(train_dataset.data[5000], cmap='gray')
    plt.title('%i' % train_dataset.targets[5000])
    plt.show()  

    img, label = train_dataset[5]
    print(img.size())
    print(label)

    # While training a model, we typically want to pass samples in “minibatches”, 
    # reshuffle the data at every epoch to reduce model overfitting.

    # Dataloader used for this





