import torch
import time
import numpy as np
from tqdm import tqdm
from random import choice
import os

from model import DDPGAgent, MADDPG

import gym
import vizdoom as vzd
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import gym_wrapper
from torch.multiprocessing import Process, Manager, Event, Queue

# 優先使用GPU資源作運算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env_name= 'MaddpgDuelFist-v0'

# 創立MADDPG架構的實例
def get_trainers(modelname,agent_num, obs_shape_n, action_shape_n):
    return MADDPG(modelname,agent_num, obs_shape_n, action_shape_n, 0.7, 20000,conv=True)

def get_act(action_n, train=True):
    #print(action_n)
    act= 0
    if train:
        act = np.random.choice(4,1,p=action_n)
    else:
        for i in range(4):
            if action_n[i]>action_n[act]:
                act = i
    #print(act)
    return act

def player1(host_arg, agent_arg, info_queue, action_queue, lock, event_obs, event_act, event_done,step_size):
    env = gym.make(env_name, host=host_arg, agent_num=agent_arg) # host參數為0意指創建本地伺服器端
    
    obs_tmp = env.reset()
    obs_tmp = obs_tmp.reshape(-1)
    info_queue.put(obs_tmp)
    event_obs.set() # 存取首次觀察資料並傳遞後，通知主程序
    
    for episode in range(0, 10001):
        # 當場景創建時，場內角色會死亡，因此必須先將其復活
        env.check_is_player_dead()
        time.sleep(1)
        for step in range(0, step_size):
            event_act.wait() # 等待主程序傳遞動作資料
            new_obs_n, rew_n, done_n, info_n = env.step(action_queue.get())
            event_act.clear() # 重置訊號
              
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1)
            # 將新的觀察資料、獎勵、完成訊號傳入主程序
            info_queue.put(new_obs_n)
            info_queue.put(rew_n[-1])
            info_queue.put(done_n[-1])
            event_obs.set() # 採取動作並完成處理後，將資料傳遞給主程序
            
            event_done.wait() # 等待父程序傳遞結束旗標
            done = info_queue.get() # 取得結束旗標
            event_done.clear()
            
            if done or step == 1199:
                obs_tmp = env.reset()
                # 當場景創建時，場內角色會死亡，因此必須先將其復活
                env.check_is_player_dead()
                time.sleep(1)
                obs_tmp = obs_tmp.reshape(-1)
                info_queue.put(obs_tmp)
                event_obs.set() # 存取觀察資料並傳遞後，通知主程序
                
                break
            
# 人類玩家模式     
def spectator():
    
    config = os.path.join(vzd.scenarios_path, "maddpg_duel_coop.cfg")
    game = vzd.DoomGame()
    game.load_config(config)
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Spectator +colorset 3")
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
    game.init()
    
    actions = [[1, 0, 0, 0, 0, 0, 0, 0]]
    for episode in range(0, 10000):
        while not game.is_episode_finished():
            if game.is_player_dead():
                game.respawn_player()
            game.make_action(choice(actions))
            
        game.new_episode()
    game.close()
                
            
    
# 主要訓練函式
#256 64
def train(update_size=256,batch_size=64,step_size=2000,agent_num=2):
    
    env = gym.make(env_name, host=1) # host參數為1意指加入本地伺服器的客戶端
    
    # for spectator mode
    if agent_num == 3:
        agent_num = agent_num - 1
    
    # 觀察空間為的高為120、寬為160、頻道數為3(RGB)
    obs_shape = []
    obs_shape.append(120)
    obs_shape.append(160)
    obs_shape.append(3)
    obs_shape_n = []  # 設定Agents觀察空間的形狀，每個Agents的觀察空間都是list中的一個元素
    action_n = 8 # 可透過print(env.action_space)得知Agent在此場景的動作空間為Discrete(8)
    action_shape_n = [] # 設定Agents動作空間的形狀
    for i in range(0, agent_num):
        obs_shape_n.append(obs_shape)
        action_shape_n.append(action_n)
        
    maddpg = get_trainers(env_name,agent_num, obs_shape_n, action_shape_n) # 創立MADDPG架構的實例
    # 初始化章節獎勵
    episode_rewards = [0.0] 
    # 每個Agent之觀察都是list obs_n中的一個元素
    obs_n = []
    obs_tmp = env.reset() # env.reset()初始化環境，並返回Agent初始觀察資料
    obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
    obs_n.append(obs_tmp)
    
    
    event_obs.wait() # 等待player1取得首次觀察資料後在執行串接
    obs_n.append(info_queue.get())
    event_obs.clear() # 將訊號重置
    epoch = 0
    for episode in tqdm(range(0, 10001)):
        # 當場景創建時，場內角色會死亡，因此必須先將其復活
        env.check_is_player_dead()
        time.sleep(1)
        for step in range(0, step_size):
            #print(step)
            # 以機率形式輸出Agents的動作
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            
            # 取得機率值最大的動作索引值，並轉換為整數資料型態
            p1_action = get_act(action_n[1],False)
            p2_action = get_act(action_n[0],False)
            action_queue.put(p1_action)
            event_act.set() # 等待action_n計算完成，再通知子程序
            
            # 返回的觀察資訊、獎勵、結束訊號、除錯資訊皆為list型態
            # 每個Agent為list中的一個元素
            new_obs_n, rew_n, done_n, info_n = env.step(p2_action)
            # 將list轉換為ndarray以執行降維
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1) # 將三維的觀察資料降成一維
            new_obs_list = []
            new_obs_list.append(new_obs_n)
            
            
            event_obs.wait() # 等待子程序傳遞資料
            new_obs_list.append(info_queue.get()) # 取得子程序資料 
            rew_n.append(info_queue.get())
            done_n.append(info_queue.get())
            event_obs.clear()
              
            # 將資訊紀錄於MADDPG結構的記憶體
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_list, done_n)
            # 將所有Agents的章節獎勵加總
            episode_rewards[-1] += np.sum(rew_n)
            
            # 每個Agent之觀察都是list obs_n中的一個元素
            obs_n = new_obs_list
            
            done = all(done_n)
            info_queue.put(done) # 將結束旗標傳遞給子程序 
            event_done.set() # 等待存取完畢，再通知子程序
            
            if step % update_size == 0:
                epoch = epoch + 1
                # 更新神經網路，目前仍在修正中，尚未完成
                x = maddpg.memory.sample(batch_size)
                maddpg.update(x)
                
            # 檢查章節是否結束
            if done or step == step_size-1:
                # 每個Agent之觀察都是list obs_n中的一個元素
                # 但目前該場景只有一個Agent，故暫時以此方式賦值
                obs_n = []
                obs_tmp = env.reset()
                # 當場景創建時，場內角色會死亡，因此必須先將其復活
                env.check_is_player_dead()
                time.sleep(1)
                obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
                obs_n.append(obs_tmp)
                
                event_obs.wait() # 等待player1取得觀察資料後，再執行串接
                obs_n.append(info_queue.get())
                event_obs.clear() # 將訊號重置
                
                episode_rewards[-1] = episode_rewards[-1]/step
                print(episode_rewards[-1])
                episode_rewards.append(0.0) # 初始化下一章節的獎勵
                break
        
        if epoch % 10 == 0:
            maddpg.update_all_agents()
        # 每隔100章節儲存一次訓練模型
        if episode % 1000 == 0:
            maddpg.save_model(episode)
            
            

def play(step_size=2000, agent_num=2):
    
    env = gym.make(env_name, host=1) # host參數為1意指加入本地伺服器的客戶端
    
    # for spectator mode
    if agent_num == 3:
        agent_num = agent_num - 1
    
    # 觀察空間為的高為120、寬為160、頻道數為3(RGB)
    obs_shape = []
    obs_shape.append(120)
    obs_shape.append(160)
    obs_shape.append(3)
    obs_shape_n = []  # 設定Agents觀察空間的形狀，每個Agents的觀察空間都是list中的一個元素
    action_n = 8 # 可透過print(env.action_space)得知Agent在此場景的動作空間為Discrete(8)
    action_shape_n = [] # 設定Agents動作空間的形狀
    for i in range(0, agent_num):
        obs_shape_n.append(obs_shape)
        action_shape_n.append(action_n)
        
    maddpg = get_trainers(env_name,agent_num, obs_shape_n, action_shape_n) # 創立MADDPG架構的實例
    maddpg.load_model(2000) # 載入訓練完成之模型
    
    # 初始化章節獎勵
    episode_rewards = [0.0] 
    # 每個Agent之觀察都是list obs_n中的一個元素
    obs_n = []
    obs_tmp = env.reset() # env.reset()初始化環境，並返回Agent初始觀察資料
    obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
    obs_n.append(obs_tmp)
    
    event_obs.wait() # 等待player1取得首次觀察資料後在執行串接
    obs_n.append(info_queue.get())
    event_obs.clear() # 將訊號重置
    epoch = 0
    for episode in tqdm(range(0, 10001)):
        # 當場景創建時，場內角色會死亡，因此必須先將其復活
        env.check_is_player_dead()
        time.sleep(1)
        for step in range(0, step_size):
            #print(step)
            # 以機率形式輸出Agents的動作
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            
            # 取得機率值最大的動作索引值，並轉換為整數資料型態
            p1_action = get_act(action_n[1],False)
            p2_action = get_act(action_n[0],False)
            action_queue.put(p1_action)
            event_act.set() # 等待action_n計算完成，再通知子程序
            
            # 返回的觀察資訊、獎勵、結束訊號、除錯資訊皆為list型態
            # 每個Agent為list中的一個元素
            new_obs_n, rew_n, done_n, info_n = env.step(p2_action)
            # 將list轉換為ndarray以執行降維
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1) # 將三維的觀察資料降成一維
            new_obs_list = []
            new_obs_list.append(new_obs_n)
                
            event_obs.wait() # 等待子程序傳遞資料
            new_obs_list.append(info_queue.get()) # 取得子程序資料 
            rew_n.append(info_queue.get())
            done_n.append(info_queue.get())
            event_obs.clear()
              
            # 將資訊紀錄於MADDPG結構的記憶體
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_list, done_n)
            # 將所有Agents的章節獎勵加總
            episode_rewards[-1] += np.sum(rew_n)
            
            # 每個Agent之觀察都是list obs_n中的一個元素
            obs_n = new_obs_list
            
            done = all(done_n)
            info_queue.put(done) # 將結束旗標傳遞給子程序 
            event_done.set() # 等待存取完畢，再通知子程序
                        
            # 檢查章節是否結束
            if done or step == step_size-1:
                # 每個Agent之觀察都是list obs_n中的一個元素
                obs_n = []
                obs_tmp = env.reset()
                # 當場景創建時，場內角色會死亡，因此必須先將其復活
                env.check_is_player_dead()
                time.sleep(1)
                obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
                obs_n.append(obs_tmp)
                
                event_obs.wait() # 等待player1取得觀察資料後，再執行串接
                obs_n.append(info_queue.get())
                event_obs.clear() # 將訊號重置
                
                episode_rewards[-1] = episode_rewards[-1]/step
                print(episode_rewards[-1])
                episode_rewards.append(0.0) # 初始化下一章節的獎勵
                break


if __name__ == '__main__':
    
    # 設定觀察者模式與遊玩模式變數
    spectator_mode = False
    play_mode = False
    
    if spectator_mode == True:
        # agent_num = 3 for spectator mode
        agent_num = 3
    else:
        agent_num = 2
        
    host_arg = 0
    
    manager = Manager()
    # 創建queue作為主程序與子程序的溝通管道
    info_queue = Queue()
    action_queue = Queue()
    # 創建lock以確保同時間只能有一個程序存取變數
    lock = manager.Lock()
    # 創建event以協助程序溝通
    event_obs = Event() # 初始化時訊號為False
    event_act = Event()
    event_done = Event()
    step_size=2000
    
    if spectator_mode == True:
        spectator_proc = Process(target=spectator)
    player1_proc = Process(target=player1, args=(host_arg, agent_num, info_queue, action_queue, lock, event_obs, event_act, event_done,step_size))
    player1_proc.start()
    if spectator_mode == True:
        spectator_proc.start()
        
    # 執行遊玩模式或訓練模式    
    if play_mode == True:
        play(step_size, agent_num)
    else:
        train(256, 64, step_size, agent_num)
        
    player1_proc.join()
    if spectator_mode == True:
        spectator_proc.join()