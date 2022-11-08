import torch.nn as nn
import torch

#torch.manual_seed(123)

class MyRNN(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.out = out
        rang = (1/out) ** 0.5

        self.g = nn.Parameter(torch.empty(inp, out))
        self.gb = nn.Parameter(torch.empty(out))
 
        self.h = nn.Parameter(torch.empty(inp, out))
        self.hb = nn.Parameter(torch.empty(out))
 
        self.r = nn.Parameter(torch.empty(1, out))
        self.rb = nn.Parameter(torch.empty(out))

        nn.init.uniform_(self.g, -rang, rang)
        nn.init.uniform_(self.gb, -rang, rang)
        nn.init.uniform_(self.h, -rang, rang)
        nn.init.uniform_(self.hb, -rang, rang)
        nn.init.uniform_(self.r, -rang, rang)
        nn.init.uniform_(self.rb, -rang, rang)

    def forward(self, x, y=None):
        y = torch.zeros(self.out) if y == None else y
        
        for n in range(len(x)):
            r = 2 * torch.sigmoid(torch.tensor([n/len(x)]) @ self.r + self.rb)
            
            g = torch.sigmoid(x[n] @ self.g + self.gb) * r
            h = torch.tanh(x[n] @ self.h + self.hb) * r

            y = (y * g) + h

        return y
        
if __name__ == '__main__':
    net = MyRNN(5, 5)
    inp = torch.randn((50, 50, 5))
    out = torch.rand((50, 5))
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=1e-2,
                                 weight_decay=1e-3)
    LF = nn.CrossEntropyLoss()
    loss_sum = 0

    net.train()
    for ep in range(10):
        for idx in range(len(inp)):
            pred = net(inp[idx])
            loss = LF(pred, out[idx]) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            loss_sum += loss.item()
            if idx%50 == 49:
                print('loss: ', round(loss_sum/50, 5))
                loss_sum = 0
