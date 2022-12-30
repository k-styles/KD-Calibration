from .resnet_cifar import _get_cifar_resnet
from .convnet_cifar import _get_cifar_convnet

# from .resnet_tinyimagenet import resnet18 as resnet18_tin
# from .resnet_tinyimagenet import resnet34 as resnet34_tin
# from .resnet_tinyimagenet import resnet50 as resnet50_tin
# from .resnet_tinyimagenet import resnet110 as resnet110_tin
# from .resnet_tinyimagenet import resnet152 as resnet152_tin

# from .wideresnet import WideResNet_40_1, WideResNet_40_2

def get_model(model, dataset, args):
    if "cifar" in dataset:
        if "resnet" in model:
            return _get_cifar_resnet(model, dataset)
        else:
            return _get_cifar_convnet(model, dataset)