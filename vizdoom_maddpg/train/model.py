from email.errors import ObsoleteHeaderDefect
from operator import truediv
import torch
from torch import nn
from memory import Memory
import torch.nn.functional as F
import os

# 優先使用GPU資源作運算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Actor被设定为一个三层全连接神经网络，输出为(-1,1)
class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_1, n_hidden_2, conv=True):
        super(Actor, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5), nn.ReLU(True))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=10, out_channels=30,kernel_size=5), nn.ReLU(True))
        #print(state_dim)
        state_dim_0 = (((state_dim[0]-4)/2-4)/2)
        state_dim_1 = (((state_dim[1]-4)/2-4)/2)
        state_dim_0 = int(state_dim_0)
        state_dim_1 = int(state_dim_1)
        self.fc1 = nn.Sequential(nn.Linear(state_dim_0*state_dim_1*30, n_hidden_1), nn.ReLU(True))
        self.fc2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.fc3 = nn.Sequential(nn.Linear(n_hidden_2, action_dim), nn.Tanh())
        self.layer1 = nn.Sequential(nn.Linear(state_dim[0]*state_dim[1]*state_dim[2], n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, action_dim), nn.Tanh())
        self.conv = conv

    def forward(self, x):
        # 檢查輸入資料之形狀
        #print('x.shape:', x.shape)
        if self.conv:
            x = torch.reshape(x, (1, 3, 120, 160))
            #print('x.shape: ', x.shape)
            #os.system("pause")
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = torch.flatten(x,1)
            #print('x=',x)
            x = self.fc1(x)
            x = self.fc2(x)
            x = self.fc3(x)
        else:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        #print('x.shape: ', x.shape)
        x = x.reshape(-1)
        #print('x:',x)
        #os.system("pause")
        return x


# Critic被设定为一个三层全连接神经网络，输出为一个linear值(这里不使用tanh函数是因为原始的奖励没有取值范围的限制)
class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, n_hidden_1, n_hidden_2):
        super(Critic, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(state_dim + action_dim, n_hidden_1), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, 1))

    def forward(self, sa):
        sa = sa.reshape(sa.size()[0], sa.size()[1] * sa.size()[2])   # 將資料降成一維的原始程式碼
        #sa = torch.flatten(sa, 0, -1) # 第二種將資料降成一維的程式碼
        x = self.layer1(sa)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# 梯度下降參數
LR_C = 1e-3
LR_A = 1e-3


# 在DDPG中，訓練網路的參數不是直接複製給目標網路的，而是軟更新的過程，也就是 v_new = (1-tau) * v_old + tau * v_new
def soft_update(net_target, net, tau):
    for target_param, param in zip(net_target.parameters(), net.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class DDPGAgent(object):

    def __init__(self, index, memory_size, batch_size, gamma, state_global, action_global, local=False, conv=False):
        self.hidden1 = 256
        self.hidden2 = 256
        self.memory = Memory(memory_size)
        self.state_dim = state_global[index]
        self.action_dim = action_global[index]
        self.Actor = Actor(self.state_dim, self.action_dim, self.hidden1, self.hidden2, conv=conv).to(device)
        # local determine one agent's information or all the agents' information, that is to say
        # it determines DDPG or MADDPG algorithm.
        if not local:
            #MADDPG
            sum_state_global=0
            for st in state_global:
                sum_state_global = sum_state_global + st[0]*st[1]*st[2]
            self.Critic = Critic(sum_state_global, sum(action_global), self.hidden1, self.hidden2).to(device)
            #self.Critic = Critic(sum(state_global), sum(action_global), self.hidden1, self.hidden2).to(device)
        else:
            #DDPG
            self.Critic = Critic(self.state_dim, self.action_dim, self.hidden1, self.hidden2).to(device)
        self.Actor_target = Actor(self.state_dim, self.action_dim, self.hidden1, self.hidden2, conv=conv).to(device)
        if not local:
            #MADDPG
            sum_state_global=0
            for st in state_global:
                sum_state_global = sum_state_global + st[0]*st[1]*st[2]
            self.Critic_target = Critic(sum_state_global, sum(action_global), self.hidden1, self.hidden2).to(device)
            #self.Critic_target = Critic(sum(state_global), sum(action_global), self.hidden1, self.hidden2).to(device)
        else:
            #DDPG
            self.Critic_target = Critic(self.state_dim, self.action_dim, self.hidden1, self.hidden2).to(device)
        self.critic_train = torch.optim.Adam(self.Critic.parameters(), lr=LR_C)
        self.actor_train = torch.optim.Adam(self.Actor.parameters(), lr=LR_A)
        self.loss_td = nn.MSELoss().to(device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = 0.5
        self.local = local

    # 輸出確切的Agent動作
    def act(self, s):
        return self.Actor(s)

    # 以機率形式輸出Agent的動作
    def act_prob(self, s):
        s = torch.flatten(s, 0, -1)
        a = self.Actor(s)
        noise = torch.normal(mean=0.0, std=torch.Tensor(size=([len(a)])).fill_(0.1)).to(device)
        a_noise = a + noise
        return a_noise


class MADDPG(object):
    def __init__(self,model_name, n, state_global, action_global, gamma, memory_size, conv=False):
        self.n = n
        self.gamma = gamma
        self.memory = Memory(memory_size)
        # 由於訓練資料大，故將Batch size設定為1
        #self.agents = [DDPGAgent(index, 1600, 1, 0.5, state_global, action_global) for index in range(0, n)]
        # alought change the batch size here, but the batch size of the epoch is determined in main.py
        self.agents = [DDPGAgent(index, 1600, 400, 0.5, state_global, action_global, conv=conv) for index in range(0, n)]
        self.model_name=model_name
        self.model_count=0
        self.conv=conv

    
    def update_agent(self, sample, index):
        observations, actions, rewards, next_obs, dones = sample
        
        # 測試用
        '''
        if(index == 0):
            print('obs:')
            print(observations)
            print('actions:')
            print(actions)
            print('rewards:')
            print(rewards)
            print('next_obs:')
            print(next_obs)
        '''
        ## if conv version, then input one photo once.
        curr_agent = self.agents[index]
        curr_agent.critic_train.zero_grad()
        all_target_actions = []
        if self.conv:
            #print('shape: ',observations.shape[0])
            for __ in range(observations.shape[0]):
                # 根據局部觀測值輸出目標動作網路的決策 action
                for i in range(0, self.n):
                    #print('next_obs:',next_obs[__, i])
                    #os.system("pause")
                    action = curr_agent.Actor_target(next_obs[__, i])
                    #print(action)
                    #os.system("pause")
                    all_target_actions.append(action)
        else:
            #print('ob:',observations.shape)
            # 根據局部觀測值輸出目標動作網路的決策 action
            for i in range(0, self.n):
                action = curr_agent.Actor_target(next_obs[:, i])
                #print(action)
                #os.system("pause")
                all_target_actions.append(action)
        ##
        #print('all_target_actions:',all_target_actions)
        action_target_all = torch.cat(all_target_actions, dim=0).to(device).reshape(actions.size()[0], actions.size()[1],
                                                                       actions.size()[2])
        target_vf_in = torch.cat((next_obs, action_target_all), dim=2)
        del action_target_all
        # calculate the value of the value of the state which the target network have.
        # 計算在目標網路下,基於Bellman Equation得到當前情況的評價
        target_value = rewards[:, index] + self.gamma * curr_agent.Critic_target(target_vf_in).squeeze(dim=1)
        del target_vf_in
        vf_in = torch.cat((observations, actions), dim=2)
        actual_value = curr_agent.Critic(vf_in).squeeze(dim=1)
        del vf_in
        # calculate the loss function to the curr_agnet.
        # 計算針對Critic的損失函數
        vf_loss = curr_agent.loss_td(actual_value, target_value.detach())
        del target_value
        del actual_value

        vf_loss.backward()
        del vf_loss
        curr_agent.critic_train.step()

        curr_agent.actor_train.zero_grad()
        ##edit here.
        if self.conv:
            for __ in range(observations.shape[0]):
                act = curr_agent.Actor_target(next_obs[__, i])
                if __ == 0:
                    curr_pol_out = act
                else:
                    curr_pol_out = torch.cat((curr_pol_out,act))
        else:
            curr_pol_out=(curr_agent.Actor(observations[:, index]))
        #print('curr_pol_out: ',curr_pol_out)
        #os.system("pause")
        curr_pol_vf_in = curr_pol_out
        all_pol_acs = []
        for i in range(0, self.n):
            if i == index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(self.agents[i].Actor(observations[:, i]).detach())
        vf_in = torch.cat((observations,
                           torch.cat(all_pol_acs, dim=0).to(device).reshape(actions.size()[0], actions.size()[1],
                                                                            actions.size()[2])), dim=2)
        # DDPG中針對Actor的損失函數
        pol_loss = -torch.mean(curr_agent.Critic(vf_in))
        pol_loss.backward()
        curr_agent.actor_train.step()
        
        
        
    def update(self, sample):
        for index in range(0, self.n):
            self.update_agent(sample, index)

    def update_all_agents(self):
        for agent in self.agents:
            soft_update(agent.Critic_target, agent.Critic, agent.tau)
            soft_update(agent.Actor_target, agent.Actor, agent.tau)

    def add_data(self, s, a, r, s_, done):
        self.memory.add(s, a, r, s_, done)

    def save_model(self, episode):
        if self.conv:
            for i in range(0, self.n):
                model_name_c = self.model_name+"_CNN_Critic_Agent" + str(i) + "_" + str(episode+self.model_count) + ".pt"
                model_name_a = self.model_name+"_CNN_Actor_Agent" + str(i) + "_" + str(episode+self.model_count) + ".pt"
                torch.save(self.agents[i].Critic_target, 'model_tag/' + model_name_c)
                torch.save(self.agents[i].Actor_target, 'model_tag/' + model_name_a)
        else:
            for i in range(0, self.n):
                model_name_c = self.model_name+"_Critic_Agent" + str(i) + "_" + str(episode+self.model_count) + ".pt"
                model_name_a = self.model_name+"_Actor_Agent" + str(i) + "_" + str(episode+self.model_count) + ".pt"
                torch.save(self.agents[i].Critic_target, 'model_tag/' + model_name_c)
                torch.save(self.agents[i].Actor_target, 'model_tag/' + model_name_a)

    def load_model(self, episode):
        self.model_count=episode
        if self.conv:
            for i in range(0, self.n):
                model_name_c = self.model_name+"_CNN_Critic_Agent" + str(i) + "_" + str(episode) + ".pt"
                model_name_a = self.model_name+"_CNN_Actor_Agent" + str(i) + "_" + str(episode) + ".pt"
                self.agents[i].Critic_target = torch.load("model_tag/" + model_name_c)
                self.agents[i].Critic = torch.load("model_tag/" + model_name_c)
                self.agents[i].Actor_target = torch.load("model_tag/" + model_name_a)
                self.agents[i].Actor = torch.load("model_tag/" + model_name_a)
        else:
            for i in range(0, self.n):
                model_name_c = self.model_name+"_Critic_Agent" + str(i) + "_" + str(episode) + ".pt"
                model_name_a = self.model_name+"_Actor_Agent" + str(i) + "_" + str(episode) + ".pt"
                self.agents[i].Critic_target = torch.load("model_tag/" + model_name_c)
                self.agents[i].Critic = torch.load("model_tag/" + model_name_c)
                self.agents[i].Actor_target = torch.load("model_tag/" + model_name_a)
                self.agents[i].Actor = torch.load("model_tag/" + model_name_a)