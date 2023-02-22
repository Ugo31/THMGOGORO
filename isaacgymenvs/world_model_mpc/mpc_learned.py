from torch import nn

class NN(nn.Module):
    def __init__(self,obs_space,action_space,time_window):
        
        super(NN, self).__init__()
        
        print("===========")
        print("obs_space*time_window")
        print(obs_space,"*",time_window)
        print("=",obs_space*time_window)
        
        self.actions_space = action_space
        self.time_window = time_window

        self.linear_stack = nn.Sequential(
                                            nn.Linear(obs_space, 128),
                                            nn.SELU(),
                                            nn.Linear(128, 256),
                                            nn.SELU(0.3),
                                            nn.Linear(256, 256),
                                            nn.SELU(0.3),
                                            nn.Linear(256, 128),
                                            nn.SELU(0.3),
                                            nn.Linear(128, action_space*time_window),
                                            nn.Tanh(),
                                        )
        

    def forward(self, input):
        return self.linear_stack(input).view(-1,self.actions_space,self.time_window)