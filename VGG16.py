
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.9),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv2
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.9),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv3
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(128, 0.9),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, 0.9),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv4
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # conv5
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
            )

        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(512, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # fc2
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            # fc3
            nn.Linear(4096, 1000),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(x), -1)
        x = self.classifier(x)
        return x

net=VGG16()
print(net)

