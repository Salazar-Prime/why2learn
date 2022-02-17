import torch
import torch.nn as nn


class TemplateNet(nn.Module):
    def __init__(self):
        super(TemplateNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3)  # (A)
        self.conv2 = nn.Conv2d(128, 128, 3)  # (B)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(XXXX, 1000)  # (C)
        self.fc2 = nn.Linear(1000, XX)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement shown below can be invoked twice with
        ## and without padding. How about three times?
        # x = self.pool(F.relu(self.conv2(x))) ## (D)
        x = x.view(-1, XXXX)  ## (E)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    raise Exception("You requested GPU support, but there's no GPU on this machine")


def run_code_for_training(net):
    net = net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_data_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 500 == 0:
                print(
                    "\n[epoch:%d, batch:%5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / float(500))
                )
                running_loss = 0.0


net = TemplateNet()
run_code_for_training(net)
