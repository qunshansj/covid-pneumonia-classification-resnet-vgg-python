
# 首先将vgg第一层卷积代码加入残差块结构
class My_Model_Blook(nn.Module):
    def __init__(self,input_channels, out_channels, kernel_size):  # 3 64 3
        super(My_Model_Blook, self).__init__()
        self.function = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm2d(out_channels, 0.9),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm2d(out_channels, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(input_channels, out_channels, kernel_size, 1, 1),
            nn.BatchNorm2d(out_channels, 0.9),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # nn.Conv2d(out_channels, out_channels, kernel_size, 1, 1),
            # nn.BatchNorm2d(out_channels, 0.9),
            # nn.ReLU(),
            # nn.MaxPool2d(2, 2)
        )


    def forward(self, x):
        identify = x
        identify = self.downsample(identify)
        f = self.function(x)

        # print(f"f:{f.size()}  identify:{identify.size()}")

        out = f + identify
        return out

# 将残差块代码封装复用
class My_Model(nn.Module):
    def __init__(self,in_channels=3, num_classes=10):
        super(My_Model, self).__init__()
        self.conv1 = nn.Sequential(
            My_Model_Blook(3,64,3)
        )
        self.conv2 = nn.Sequential(
            My_Model_Blook(64, 128, 3)
        )
        self.conv3 = nn.Sequential(
            My_Model_Blook(128, 256, 3)
        )
        self.conv4 = nn.Sequential(
            My_Model_Blook(256, 512, 3)
        )
        
        # bug:输入输出一致时不能用1*1卷积，会发生维度消失，无法衔接后面模型
        # 解决方法：在残差块中不加1*1卷积或者单独卷积
        
        # self.conv5 = nn.Sequential(
        #     My_Model_Blook(512, 512, 3)
        # )


        self.classifier = nn.Sequential(
            # fc1
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(),
            # fc2
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(),
            # fc3
            nn.Linear(1024, num_classes),
        )
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        
        # bug:输入输出一致时不能用1*1卷积，会发生维度消失，无法衔接后面模型
        
        # print(out.size())
        # out = self.conv5(out)
        # print(out.size())
        # out = out.resize(512,1,1)

        out = out.view((x.shape[0], -1))  # 拉成一维
        out = self.classifier(out)
        return out
