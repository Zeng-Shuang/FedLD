import torch
from options import args_parser
from torchvision.models.resnet import ResNet, Bottleneck
args = args_parser()

class ResNet50(ResNet):
    def __init__(self):
        super(ResNet50, self).__init__(Bottleneck, [3, 4, 6, 3])

        #self.fc = nn.Linear(2048, 2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        y = self.fc(x)

        return x,y


