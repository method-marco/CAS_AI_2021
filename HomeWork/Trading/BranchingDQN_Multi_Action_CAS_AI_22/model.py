import torch 
import torch.nn as nn


class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 4*128),
                                   nn.ReLU(), 
                                   nn.Linear(4*128,128),
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_head = nn.Linear(128, ac)

    def forward(self, x): 

        out = self.model(x)

        value = self.value_head(out)
        adv = self.adv_head(out)

        q_val = value + adv - adv.mean(1).reshape(-1,1)
        return q_val


class BranchingQNetwork(nn.Module):

    def __init__(self, obs, ac_dim, n): 

        super().__init__()

        self.ac_dim = ac_dim # anzahl haupt-actions LunarLanderContinuous-v2 - 2
        self.n = n # bins

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(),
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(ac_dim)])
        #print(self.model)

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out) # state value
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )

        return q_val

# b = BranchingQNetwork(5, 4, 6)

# b(torch.rand(10, 5))