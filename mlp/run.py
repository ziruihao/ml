import torchvision
import torch
from mlp import MLP
from slp import SLP
from train import train
from test import test
import csv

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.5,), (0.5,))])

train_data = torchvision.datasets.MNIST('PATH_TO_STORE_TRAINSET', download=True, train=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.MNIST('PATH_TO_STORE_TESTSET', download=True, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

def train_all():

    # NLLLoss and LogSoftmax

    model = MLP(784, 10, [300])
    print('training for 3 epochs')
    train('MLP_1_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_1_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_1_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    model = MLP(784, 10, [400, 200])
    print('training for 3 epochs')
    train('MLP_2_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_2_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_2_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    model = MLP(784, 10, [600, 500, 400, 300, 200, 100])
    print('training for 3 epochs')
    train('MLP_6_NLL_SGD_3', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_6_NLL_SGD_10', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_6_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    # CrossEntropyLoss and Softmax

    model = MLP(784, 10, [300])
    print('training for 3 epochs')
    train('MLP_1_CEL_SGD_3', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_1_CEL_SGD_10', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_1_CEL_SGD_30', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    model = MLP(784, 10, [400, 200])
    print('training for 3 epochs')
    train('MLP_2_CEL_SGD_3', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_2_CEL_SGD_10', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_2_CEL_SGD_30', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    model = MLP(784, 10, [600, 500, 400, 300, 200, 100])
    print('training for 3 epochs')
    train('MLP_6_CEL_SGD_3', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 3, transform, train_data, train_loader)
    print('training for 10 epochs')
    train('MLP_6_CEL_SGD_10', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 10, transform, train_data, train_loader)
    print('training for 30 epochs')
    train('MLP_6_CEL_SGD_30', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)

    # 100 epochs with 8 hidden layers

    model = MLP(784, 10, [64, 64, 64, 64, 64, 64, 64, 64])
    print('training for 100 epochs')
    train('MLP_8_CEL_SGD', model, torch.nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 100, transform, train_data, train_loader)

def test_all():

    with open('performance.csv', mode='w') as performance:

        write = csv.writer(performance, delimiter=',', quotechar='"')

        # NLLLoss and LogSoftmax

        # model = MLP(784, 10, [300])
        # print('testing for 3 epochs')
        # write.writerow(['MLP_1_NLL_SGD_3', test('MLP_1_NLL_SGD_3', model, test_data, test_loader)])
        # print('testing for 10 epochs')
        # write.writerow(['MLP_1_NLL_SGD_10', test('MLP_1_NLL_SGD_10', model, test_data, test_loader)])
        # print('testing for 30 epochs')
        # write.writerow(['MLP_1_NLL_SGD_30', test('MLP_1_NLL_SGD_30', model, test_data, test_loader)])

        # model = MLP(784, 10, [400, 200])
        # print('testing for 3 epochs')
        # write.writerow(['MLP_2_NLL_SGD_3', test('MLP_2_NLL_SGD_3', model, test_data, test_loader)])
        # print('testing for 10 epochs')
        # write.writerow(['MLP_2_NLL_SGD_10', test('MLP_2_NLL_SGD_10', model,test_data, test_loader)])
        # print('testing for 30 epochs')
        # write.writerow(['MLP_2_NLL_SGD_30', test('MLP_2_NLL_SGD_30', model, test_data, test_loader)])

        # model = MLP(784, 10, [600, 500, 400, 300, 200, 100])
        # print('testing for 3 epochs')
        # write.writerow(['MLP_6_NLL_SGD_3', test('MLP_6_NLL_SGD_3', model, test_data, test_loader)])
        # print('testing for 10 epochs')
        # write.writerow(['MLP_6_NLL_SGD_10', test('MLP_6_NLL_SGD_10', model, test_data, test_loader)])
        # print('testing for 30 epochs')
        # write.writerow(['MLP_6_NLL_SGD_30', test('MLP_6_NLL_SGD_30', model, test_data, test_loader)])

        # # CrossEntropyLoss and Softmax

        # model = MLP(784, 10, [300])
        # print('testing for 3 epochs')
        # write.writerow(['MLP_1_CEL_SGD_3', test('MLP_1_CEL_SGD_3', model, test_data, test_loader)])
        # print('testing for 10 epochs')
        # write.writerow(['MLP_1_CEL_SGD_10', test('MLP_1_CEL_SGD_10', model, test_data, test_loader)])
        # print('testing for 30 epochs')
        # write.writerow(['MLP_1_CEL_SGD_30', test('MLP_1_CEL_SGD_30', model, test_data, test_loader)])

        # model = MLP(784, 10, [400, 200])
        # print('testing for 3 epochs')
        # write.writerow(['MLP_2_CEL_SGD_3', test('MLP_2_CEL_SGD_3', model, test_data, test_loader)])
        # print('testing for 10 epochs')
        # write.writerow(['MLP_2_CEL_SGD_10', test('MLP_2_CEL_SGD_10', model, test_data, test_loader)])
        # print('testing for 30 epochs')
        # write.writerow(['MLP_2_CEL_SGD_30', test('MLP_2_CEL_SGD_30', model, test_data, test_loader)])

        model = MLP(784, 10, [600, 500, 400, 300, 200, 100])
        # # print('testing for 3 epochs')
        write.writerow(['MLP_6_CEL_SGD_3', test('MLP_6_CEL_SGD_3', model, test_data, test_loader)])
        # # print('testing for 10 epochs')
        write.writerow(['MLP_6_CEL_SGD_10', test('MLP_6_CEL_SGD_10', model, test_data, test_loader)])
        # print('testing for 30 epochs')
        write.writerow(['MLP_6_CEL_SGD_30', test('MLP_6_CEL_SGD_30', model, test_data, test_loader)])

        # 100 epochs with 8 hidden layers

        # model = MLP(784, 10, [64, 64, 64, 64, 64, 64, 64, 64])
        # print('testing for 100 epochs')
        # for epoch in range(100):
        #     if ((epoch % 5) is 0):
        #         write.writerow(['MLP_8_CEL_SGD_' + str(epoch), test('MLP_8_CEL_SGD_' + str(epoch), model, test_data, test_loader)])

    performance.close()

model = SLP(784, 10)
train('SLP_NLL_SGD_30', model, torch.nn.NLLLoss(), torch.optim.SGD(model.parameters(), lr = 0.001), 30, transform, train_data, train_loader)
test('SLP_NLL_SGD_30', model, test_data, test_loader)