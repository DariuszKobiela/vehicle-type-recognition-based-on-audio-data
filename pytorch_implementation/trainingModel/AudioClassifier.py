import torch.nn.functional as F
from torch.nn import init
from torch import nn

# ----------------------------
# Audio Classification Model
# ----------------------------
class AudioClassifier (nn.Module):
    # ----------------------------
    # Build the model architecture
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        
        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        relu1 = nn.ReLU()
        bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(conv1.weight, a=0.1)
        conv1.bias.data.zero_()
        conv_layers += [conv1, relu1, bn1]

        # 2nd Convolution Block
        conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu2 = nn.ReLU()
        bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(conv2.weight, a=0.1)
        conv2.bias.data.zero_()
        conv_layers += [conv2, relu2, bn2]

        # 3rd Convolution Block
        conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu3 = nn.ReLU()
        bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(conv3.weight, a=0.1)
        conv3.bias.data.zero_()
        conv_layers += [conv3, relu3, bn3]

        # 4th Convolution Block
        conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        relu4 = nn.ReLU()
        bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(conv4.weight, a=0.1)
        conv4.bias.data.zero_()
        conv_layers += [conv4, relu4, bn4]

        # Linear Classifier
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=3)
        # self.lin2 = nn.Linear(in_features=20, out_features=50)
        # self.lin3 = nn.Linear(in_features=50, out_features=10)

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        
        # Autoencoder architecture
        # self.encoder = nn.Sequential(
        #     nn.Linear(469, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 36),
        #     nn.ReLU(),
        #     nn.Linear(36, 18),
        #     nn.ReLU(),
        #     nn.Linear(18, 9)
        # )
         
        # # Building an linear decoder with Linear
        # # layer followed by Relu activation function
        # # The Sigmoid activation function
        # # outputs the value between 0 and 1
        # # 9 ==> 784
        # self.decoder = nn.Sequential(
        #     nn.Linear(9, 18),
        #     nn.ReLU(),
        #     nn.Linear(18, 36),
        #     nn.ReLU(),
        #     nn.Linear(36, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 3),
        #     nn.ReLU()
        # )

 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.lin(x)

        # x = F.relu(x)
        # x = self.lin2(x)
        
        # x = F.relu(x)
        # x = self.lin3(x)
        
        # encoded = self.encoder(x)
        # decoded = self.decoder(encoded)

        # Final output
        return x
        # return decoded