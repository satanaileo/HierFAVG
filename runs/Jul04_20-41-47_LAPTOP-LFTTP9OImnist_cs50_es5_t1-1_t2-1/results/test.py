import os

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from torchvision import datasets
from hierfavg import fast_all_clients_test
from models.mnist_cnn import mnist_lenet


def main():
    params = torch.load("params.pt")
    model = mnist_lenet(1, 10)
    model.load_state_dict(params)
    model.cuda(0)
    model.train(False)
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # print(os.path.abspath(os.path.join(os.path.join(os.getcwd(),"../../../"), "data", "mnist")))
    test = datasets.MNIST(os.path.join(os.getcwd(), "../../../", "data", "mnist"), train=False,
                          download=False, transform=transform)
    v_test_loader = DataLoader(test, batch_size=20 * 50,
                               shuffle=False, **kwargs)
    correct_all_v, total_all_v = fast_all_clients_test(v_test_loader, model, torch.device('cuda:0'))
    avg_acc_v = correct_all_v / total_all_v
    print(f"The accuracy of the saved model is {avg_acc_v}")


if __name__ == '__main__':
    main()
