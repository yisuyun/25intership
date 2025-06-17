import torch
from torch import nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):  
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 64, kernel_size=5, padding=2), 
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第二层卷积
            DepthwiseSeparableConv(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # 第三层卷积
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(384, 256, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True),
            
            # 第五层卷积
            DepthwiseSeparableConv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d((6, 6))
        
        # 分类器部分
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 2048),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),  
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),  
        )

        # 初始化权重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)  
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight) 
                nn.init.constant_(m.bias, 0.1)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, 
            padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

if __name__ == '__main__':
    model = AlexNet(num_classes=1000)  
    input_tensor = torch.randn(64, 3, 32, 32) 
    output = model(input_tensor)
    print(f"输入形状: {input_tensor.shape}")
    print(f"输出形状: {output.shape} (每个样本1003维)")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
