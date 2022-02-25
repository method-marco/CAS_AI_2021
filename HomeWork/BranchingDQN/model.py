import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.distributions import Categorical 


class DuelingNetwork(nn.Module): 

    def __init__(self, obs, ac): 

        super().__init__()

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(), 
                                   nn.Linear(128,128), 
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

        self.ac_dim = ac_dim
        self.n = n 

        self.model = nn.Sequential(nn.Linear(obs, 128), 
                                   nn.ReLU(),
                                   nn.Linear(128,128), 
                                   nn.ReLU())

        self.value_head = nn.Linear(128, 1)
        self.adv_heads = nn.ModuleList([nn.Linear(128, n) for i in range(ac_dim)])

    def forward(self, x): 

        out = self.model(x)
        value = self.value_head(out)
        advs = torch.stack([l(out) for l in self.adv_heads], dim = 1)

        # print(advs.shape)
        # print(advs.mean(2).shape)
        test =  advs.mean(2, keepdim = True)
        # input(test.shape)
        q_val = value.unsqueeze(2) + advs - advs.mean(2, keepdim = True )
        # input(q_val.shape)

        return q_val

    def update_model_mixed(self, other_model, tau=0.1):
        new_target_dict = {}
        my_dict = self.state_dict()
        other_dict = other_model.state_dict()
        for param in my_dict:
            target_ratio = (1.0 - tau) * my_dict[param] #target_model
            online_ratio = tau * other_dict[param] # online_model
            mixed_weights = target_ratio + online_ratio
            new_target_dict[param] = mixed_weights
        self.load_state_dict(new_target_dict)

# b = BranchingQNetwork(5, 4, 6)

# b(torch.rand(10, 5))