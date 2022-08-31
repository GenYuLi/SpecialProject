import torch
import time
import numpy as np

from model import DDPGAgent, MADDPG

import gym
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import gym_wrapper
from torch.multiprocessing import Process, Manager

# 優先使用GPU資源作運算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 創立MADDPG架構的實例
def get_trainers(agent_num, obs_shape_n, action_shape_n):
    return MADDPG(agent_num, obs_shape_n, action_shape_n, 0.7, 20000)
    
# 主要訓練函式
def train():
    # host參數為-1(預設)意指執行單人模式場景(只有一個Agent)
    env = gym.make('VizdoomBasic-v0', host=-1)
    
    # 設定Agents觀察空間的形狀，每個Agents的觀察空間都是list中的一個元素
    obs_shape_n = []
    # 觀察空間為的高為120、寬為160、頻道數為3(RGB)
    obs_shape_n.append(120*160*3)
    # 設定Agents動作空間的形狀
    action_shape_n = []
    # 可透過print(env.action_space)得知Agent在此場景的動作空間為Discrete(4)
    action_shape_n.append(4)
    # 創立MADDPG架構的實例
    maddpg = get_trainers(agent_num, obs_shape_n, action_shape_n)
    # 初始化章節獎勵
    episode_rewards = [0.0]
    # 每個Agent之觀察都是list obs_n中的一個元素
    # 但目前該場景只有一個Agent，故暫時以此方式賦值
    obs_n = []
    # env.reset()函式將初始化環境，並返回Agent的第一個觀察資料
    obs_tmp = env.reset()
    # 將三維的觀察資料降成一維
    obs_tmp = obs_tmp.reshape(-1)
    obs_n.append(obs_tmp)
    
    for episode in range(0, 100):
        for step in range(0, 1000):
            # 以機率形式輸出Agents的動作
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            # 取得機率值最大的動作索引值，並轉換為整數資料型態
            action = int(np.argmax(action_n[0]))
            # 返回的觀察資訊、獎勵、結束訊號、除錯資訊皆為list型態
            # 每個Agent為list中的一個元素
            new_obs_n, rew_n, done_n, info_n = env.step(action)
            # 將list轉換為ndarray以執行降維
            new_obs_n = np.array(new_obs_n)
            # 將三維的觀察資料降成一維
            new_obs_n = new_obs_n.reshape(-1)
            
            new_obs_list = []
            new_obs_list.append(new_obs_n)
            # 將資訊紀錄於MADDPG結構的記憶體
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_list, done_n)
            # 將所有Agents的章節獎勵加總
            episode_rewards[-1] += np.sum(rew_n)
            
            # 每個Agent之觀察都是list obs_n中的一個元素
            # 但目前該場景只有一個Agent，故暫時以此方式賦值
            obs_n = []
            obs_n.append(new_obs_n)
            
            done = all(done_n)
            if step % 300 == 0:
                # 更新神經網路，目前仍在修正中，尚未完成
                maddpg.update(maddpg.memory.sample(batch_size))
                maddpg.update_all_agents()
            # 檢查章節是否結束
            if done or step == 999:
                # 每個Agent之觀察都是list obs_n中的一個元素
                # 但目前該場景只有一個Agent，故暫時以此方式賦值
                obs_n = []
                obs_tmp = env.reset()
                # 將三維的觀察資料降成一維
                obs_tmp = obs_tmp.reshape(-1)
                obs_n.append(obs_tmp)
                
                episode_rewards[-1] = episode_rewards[-1]/step
                print(episode_rewards[-1])
                # 初始化下一章節的獎勵
                episode_rewards.append(0.0)
                break
            
        # 每隔100章節儲存一次訓練模型
        if episode % 100 == 0:
            maddpg.save_model(episode)

# 以訓練完成的模型執行程式，暫未處理
'''
def play():
    env = make_env("simple_tag")
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    max_ob_dim = max(obs_shape_n)
    for i in range(0, len(obs_shape_n)):
        obs_shape_n[i] = max_ob_dim
    action_shape_n = [env.action_space[i].n for i in range(env.n)]
    maddpg = get_trainers(env, obs_shape_n, action_shape_n)
    maddpg.load_model(100)
    obs_n = env.reset()
    for i in range(0, len(obs_n)):
        while len(obs_n[i]) < obs_shape_n[i]:
            obs_n[i] = np.append(obs_n[i], np.array([0.0]))
    env.render()
    for episode in range(0, 10000):
        for step in range(0, 2000):
            action_n = [agent.Actor(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            print(rew_n)
            #time.sleep(0.1)
            env.render()
            for i in range(0, len(new_obs_n)):
                while len(new_obs_n[i]) < obs_shape_n[i]:
                    new_obs_n[i] = np.append(new_obs_n[i], np.array([0.0]))
            obs_n = new_obs_n
            done = all(done_n)
            if done or step == 9999:
                obs_n = env.reset()
                for i in range(0, len(obs_n)):
                    while len(obs_n[i]) < obs_shape_n[i]:
                        obs_n[i] = np.append(obs_n[i], np.array([0.0]))
                break
'''

if __name__ == '__main__':
    agent_num = 1
    batch_size = 1600
    
    train()
    #play()
    print('finish!')
