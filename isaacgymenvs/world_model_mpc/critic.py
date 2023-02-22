from torch import nn
class Critic(nn.Module):
    def __init__(self,obs_space, time_window):
        
        self.obs_space=obs_space
        self.time_window=time_window

        
        super(Critic, self).__init__()
        self.linear_stack = nn.Sequential(
                                            nn.Linear(obs_space*time_window, 128),
                                            nn.SELU(0.3),
                                            nn.Linear(128, 256),
                                            nn.SELU(0.3),
                                            nn.Linear(256, 128),
                                            nn.SELU(0.3),
                                            nn.Linear(128, time_window),
                                        )
        

    def forward(self, input):
        x = input.reshape(input.shape[0],input.shape[1]*input.shape[2])
        return self.linear_stack(x).view(input.shape[0],1,self.time_window)