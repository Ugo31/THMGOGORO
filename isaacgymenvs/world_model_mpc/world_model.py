from torch import nn
import torch
class WorldModel(nn.Module):
    def __init__(self,obs_space,action_space,time_window):
        
        super(WorldModel, self).__init__()
        
        self.obs_space=obs_space
        self.action_space=action_space
        self.time_window=time_window
        print("world model is excpecting input of size :")
        print("(obs_space+action_space)*time_window")
        print("",obs_space,"+",action_space,"*",time_window)
        print("= ",obs_space+action_space*time_window)

        self.linear_stack = nn.Sequential(
                                            nn.Linear( obs_space+action_space , 128),
                                            nn.SELU(0.3),
                                            nn.Linear(128, 512),
                                            nn.SELU(0.3),
                                            nn.Linear(512, 512),
                                            nn.SELU(0.3),
                                            nn.Linear(512, 128),
                                            nn.SELU(0.3),
                                            nn.Linear(128, obs_space),
                                        )
        

    def forward(self, x,y):
        input  = torch.cat((x, y), 1)
        output = self.linear_stack(input)
        return output
    
