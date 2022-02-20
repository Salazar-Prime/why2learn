import torch.nn as nn
import torch.nn.functional as F

class mynet(nn.Module):
    def __init__(self, task):
        super(mynet, self).__init__()
        self.task = task
        self.conv1 = nn.Conv2d(3, 128, 3)  # (A)
        self.conv2 = nn.Conv2d(128, 128, 3)  # (B)
        self.pool = nn.MaxPool2d(2, 2)
        # Changing line (C)
        if self.task==1:
            self.fc1 = nn.Linear(128*31*31, 1000)  # (C)
        elif self.task==2:
            self.fc1 = nn.Linear(128*14*14, 1000) 
        else:
            self.conv1 = nn.Conv2d(3, 128, 3, padding=1)
            self.fc1 = nn.Linear(128*15*15, 1000) 
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        ## Uncomment the next statement and see what happens to the
        ## performance of your classifier with and without padding.
        ## Note that you will have to change the first arg in the
        ## call to Linear in line (C) above and in the line (E)
        ## shown below. After you have done this experiment, see
        ## if the statement show n below can be invoked twice with
        ## and without padding. How about three times?
        
        # Changing line (E)
        if self.task==1:
            x = x.view(-1, 128*31*31)
        elif self.task==2:
            x = self.pool(F.relu(self.conv2(x))) ## (D)
            x = x.view(-1, 128*14*14)
        else:
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 128*15*15)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x