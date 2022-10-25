from torch import nn
from torch.nn import init

def weight_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()
            #init.xavier_uniform(m.weight.data)
            #init.constant_(m.bias.data,0.1)
        elif isinstance(m, nn.BatchNorm2d):
           # m.weight.data.fill_(1)
            m.weight.data.normal_(0,0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()