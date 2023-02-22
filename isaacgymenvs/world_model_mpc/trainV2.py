
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
n_envs = 1024
envs = isaacgymenvs.make(
                            seed=0, 
                            task="Cartpole", 
                            num_envs=n_envs, 
                            sim_device=device,
                            rl_device=device,
                            graphics_device_id=0,
                            headless=False
                        )

N = 10
WM_TRAINING = 1000

wm = WorldModel(envs.observation_space.shape[0],
                envs.action_space.shape[0],
                N).to(device)

nn = NN(envs.observation_space.shape[0],
        envs.action_space.shape[0],
        N).to(device)

cr = Critic(envs.observation_space.shape[0],
            N).to(device)


def rnn_frwd(obs, action_seq, real_obs):
    
    predicted_ = torch.zeros((n_envs,envs.observation_space.shape[0],N),device=device)
    wm.zero_grad()
    pred_obs = obs
    world_loss = 0.0
    for r in range(action_seq.shape[2]):
        pred_obs    = wm.forward(action_seq[:,:,r],pred_obs)
        if(real_obs is not None):
            world_loss += (torch.sum(real_obs[:,:,r]-pred_obs)/n_envs)**2
        predicted_[:,:,r] = pred_obs

    return world_loss,predicted_



obs = envs.reset()
previous_obs = obs['obs']


N_action_tensor    = torch.zeros((n_envs,envs.action_space.shape[0],N+1), device=device)
N_obs_tensor       = torch.zeros((n_envs,envs.observation_space.shape[0],N+1), device=device)
N_reward_tensor    = torch.zeros((n_envs,1,N+1), device=device)
N_next_obs_tensor  = torch.zeros((n_envs,1,N+1), device=device)

world_optimizer   = torch.optim.Adam(wm.parameters(), lr=0.00001)
critic_optimizer  = torch.optim.Adam(cr.parameters(), lr=0.00001)
actor_optimizer   = torch.optim.Adam(cr.parameters(), lr=0.001)

curent_step = 0

action_loss = torch.zeros(1).to(device)

curent_action = torch.rand((n_envs,envs.action_space.shape[0]), device=device)*2-1
while True:
    
    print("==========")
    
    if(curent_step < WM_TRAINING):
        curent_action = torch.rand((n_envs,envs.action_space.shape[0]), device=device)*2-1


    obs, rwd , done, info = envs.step(curent_action)
    curent_obs            = obs['obs']
    
    N_obs_tensor[:,:,:-1] = N_obs_tensor[:,:,1:].clone()
    N_obs_tensor[:,:,-1]  = previous_obs
    
    N_action_tensor[:,:,:-1] = N_action_tensor[:,:,1:].clone()
    N_action_tensor[:,:,-1]  = curent_action

    N_reward_tensor[:,:,:-1] = N_reward_tensor[:,:,1:].clone()
    N_reward_tensor[:,0,-1]  = rwd




    obs_0   = N_obs_tensor[:,:,0].clone().detach()
    obs_1_N = N_obs_tensor[:,:,1:].clone().detach()
    obs_0_N_1 = N_obs_tensor[:,:,:-1].clone().detach()

    act_0 = N_action_tensor[:,:,0].clone().detach()
    act_1_N = N_action_tensor[:,:,1:].clone().detach()
    act_0_N_1 = N_action_tensor[:,:,0:-1].clone().detach()

    rwd_1_N = N_action_tensor[:,:,1:].clone().detach()


    use_grad(wm,True)
    use_grad(cr,True)
    
    # curent_obs      = curent_obs.clone()
    # curent_obs = curent_obs.detach()
    # curent_obs.requires_grad = False
    
    #Training the world model and critic
    # #===========================================================
    
    #INPUT :
    #
    # THE CURENT STATE    (nenvs,state_shape)
    # SEQUENCE PLANED ACTIONS  (nenvs,action_shape, horizon_len)
    #
    #RETURN :
    #
    # SEQUENCE OF FUTURES STATES (nenvs,state_shape, horizon_len)
    
    world_loss,predictions = rnn_frwd(obs_0, act_0_N_1, obs_1_N)
    world_optimizer.zero_grad()
    world_loss.backward()
    world_optimizer.step()
    # N_pred_obs = wm.forward(act_0_N_1,obs_0) 
    # world_loss  = (torch.sum(obs_1_N-N_pred_obs)/n_envs)**2
    # world_optimizer.zero_grad()
    # world_loss.backward()
    # world_optimizer.step()

    # print("====")
    # print("obs_1_N=",obs_1_N[0])
    # print("N_pred_obs=",predictions[0])

    #===========================================================
    #INPUT :
    #
    # SEQUENCE OF OBS
    #
    #OUTPUT
    #
    # SEQUENCE OF REWARDS
    
    N_pred_rwd = cr.forward(obs_0_N_1)
    critic_loss      = (torch.sum(rwd_1_N-N_pred_rwd)/n_envs)**2
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    

    #==========================================================
    #if the observation is considered new we train the network
    if(world_loss>0.3 and curent_step > WM_TRAINING):
        use_grad(wm,False)
        use_grad(cr,False)
        use_grad(nn,False)
        
        #a = nn.forward(curent_obs)
        a = nn.forward(curent_obs)#torch.rand(n_envs,envs.action_space.shape[0],N,device=device,requires_grad = True)
        optimised_action_seq = Variable(a.clone().detach())
        optimised_action_seq.requires_grad = True
        action_optimizer  = torch.optim.Adam([optimised_action_seq] , lr=0.01)


        print("=====")
        print(optimised_action_seq.shape)
        for y in range(100):
            #predict_obs_seq    = wm.forward(optimised_action_seq,curent_obs)
            
            _,predict_obs_seq= rnn_frwd(curent_obs, optimised_action_seq, None)
            
            predict_reward = cr.forward(predict_obs_seq)
            
            action_loss    = torch.sum(-predict_reward)/n_envs #TODO add delta penalisation

            action_optimizer.zero_grad()
            action_loss.backward()
            action_optimizer.step()

            # if(y == 0):
            #     print("before opt =",action_loss.item())
            #     print("before opt =",torch.sum(predict_reward).item())
            # if(y == 99):
            #     print("after  opt =",action_loss.item())
            #     print("after  opt =",torch.sum(predict_reward).item())

        use_grad(nn,True)
        generated_action_seq = optimised_action_seq.clone().detach()
        generated_action_seq.requires_grad = False
        predicted_action_seq = nn.forward(curent_obs)
        actor_loss      = (torch.sum(predicted_action_seq-generated_action_seq)/n_envs)**2
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
    else:
        use_grad(nn,False)
        generated_action_seq = nn.forward(curent_obs)
        
        
    curent_action = generated_action_seq[:,0,0].clone().view(n_envs,envs.action_space.shape[0])

    print(curent_step,"| WORLD LOSS =",world_loss.item(), "| CRITIC LOSS = ",critic_loss.item() , "| ACTOR LOSS = ",action_loss.item())
    
    previous_obs = curent_obs
    curent_step+=1
    
