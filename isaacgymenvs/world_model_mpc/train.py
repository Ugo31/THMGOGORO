
import isaacgym
import isaacgymenvs
import torch
from world_model import WorldModel
from mpc_learned import NN
from critic import Critic
import torch.optim as optim
from torch.autograd import Variable

def use_grad(model,bool_):
    for param in model.parameters():
        param.requires_grad = bool_



device = "cuda:0"
n_envs = 64
envs = isaacgymenvs.make(
                            seed=0, 
                            task="Cartpole", 
                            num_envs=n_envs, 
                            sim_device=device,
                            rl_device=device,
                            graphics_device_id=0,
                            headless=False
                        )

wm = WorldModel(envs.observation_space.shape[0],envs.action_space.shape[0],1).to(device)
nn = NN(envs.observation_space.shape[0],envs.action_space.shape[0]).to(device)
cr = Critic(envs.observation_space.shape[0]).to(device)

obs = envs.reset()
previous_obs = obs['obs']


prev_action = torch.rand((n_envs,)+envs.action_space.shape, device=device)*2-1

world_optimizer   = torch.optim.Adam(wm.parameters(), lr=0.00001)
critic_optimizer  = torch.optim.Adam(cr.parameters(), lr=0.0001)
actor_optimizer   = torch.optim.Adam(cr.parameters(), lr=0.0001)

curent_step = 0
while True:
    
    print("==========")
    
    
    
    
    obs, rwd , done, info = envs.step(prev_action)
    obs= obs['obs']

    avg_loss_world = 0
    avg_loss_critic = 0
    avg_actor_loss = torch.zeros(1).to(device)

    

    for i in range(n_envs):
        
        use_grad(wm,True)
        use_grad(cr,True)
        
        action   = prev_action[i].clone()
        prev_obs = previous_obs[i].clone()
        reward   = rwd[i].clone()
        reel_obs = obs[i].clone()
        
        action   = action.detach()
        prev_obs = prev_obs.detach()
        reward   = reward.detach()
        reel_obs = reel_obs.detach()

        
        
        action.requires_grad = False
        prev_obs.requires_grad = False
        reel_obs.requires_grad = False
        
        #Training the world model and critic
        # #===========================================================
        pred_obs = wm.forward(action,prev_obs)
        world_loss  = torch.sum(reel_obs-pred_obs)**2
        world_optimizer.zero_grad()
        world_loss.backward()
        world_optimizer.step()
        avg_loss_world+= world_loss
        
        #===========================================================        
        predicted_reward = cr.forward(reel_obs)
        critic_loss      = torch.sum(reward-predicted_reward)**2
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        avg_loss_critic += critic_loss
        #==========================================================
        #if the observation is considered new we train the network
        if(world_loss>0.03 and curent_step > 1000):
            use_grad(wm,False)
            use_grad(cr,False)
            
            input_action = Variable(torch.rand(envs.action_space.shape[0],device=device))
            input_action.requires_grad = True
            action_optimizer  = torch.optim.SGD([input_action] , lr=0.01)
            for y in range(1000):
                predict_obs    = wm.forward(input_action,prev_obs)
                predict_reward = cr.forward(predict_obs)

                action_loss    = -predict_reward

                action_optimizer.zero_grad()
                action_loss.backward()
                action_optimizer.step()
                
            input_action.requires_grad = False
            predicted_action = nn.forward(reel_obs)
            actor_loss      = torch.sum(predicted_action-input_action)**2
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()
            avg_actor_loss += actor_loss

                #print("input_action[i] =",input_action.item(),"\t final prediction =",predict_reward.item())
        else:


            input_action = nn.forward(prev_obs)
            
            
            
            
        
        prev_action[i] = input_action.clone()
    
    print(curent_step,"| CRITIC LOSS =",(avg_loss_critic/n_envs).item(), "| WORLD LOSS = ",(avg_loss_critic/n_envs).item() , "| ACTOR LOSS = ",(avg_actor_loss/n_envs).item())
    
    previous_obs = obs
    curent_step+=1
