import torch
import time
import numpy as np

from model import DDPGAgent, MADDPG

import gym
from vizdoom.gym_wrapper.base_gym_env import VizdoomEnv
from vizdoom import gym_wrapper
from torch.multiprocessing import Process, Manager, Event, Queue

# 優先使用GPU資源作運算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"

# 創立MADDPG架構的實例
def get_trainers(obs_shape_n, action_shape_n):
    return MADDPG(2, obs_shape_n, action_shape_n, 0.7, 20000)

def player1(info_queue, action_queue, lock, event_obs, event_act, event_done):
    env = gym.make('VizdoomMultipleInstances-v0', host=1)
    obs_tmp = env.reset()
    obs_tmp = obs_tmp.reshape(-1)
    info_queue.put(obs_tmp)
    event_obs.set() # 存取首次觀察資料並傳遞後，通知主程序
    
    for episode in range(0, 10000):
        for step in range(0, 2000):
            event_act.wait() # 等待主程序傳遞動作資料
            new_obs_n, rew_n, done_n, info_n = env.step(action_queue.get())
            event_act.clear() # 重置訊號
              
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1)
            # 將新的觀察資料、獎勵、完成訊號傳入主程序
            #lock.acquire()
            info_queue.put(new_obs_n)
            info_queue.put(rew_n[-1])
            info_queue.put(done_n[-1])
            #lock.release()
            event_obs.set() # 採取動作並完成處理後，將資料傳遞給主程序
            
            event_done.wait() # 等待父程序傳遞結束旗標
            #lock.acquire()
            done = info_queue.get() # 取得結束旗標
            #lock.release()
            event_done.clear()
            
            if done or step == 1999:
                obs_tmp = env.reset()
                obs_tmp = obs_tmp.reshape(-1)
                info_queue.put(obs_tmp)
                event_obs.set() # 存取觀察資料並傳遞後，通知主程序
                
                break
                
            
    
# 主要訓練函式
def train():
    
    agent_n = 2 # 該場景的agents數量
    env = gym.make('VizdoomMultipleInstances-v0', host=0) # host參數為2意指執行單人模式場景(只有一個Agent)
    
    obs_shape = 120*160*3 # 觀察空間為的高為120、寬為160、頻道數為3(RGB)
    obs_shape_n = []  # 設定Agents觀察空間的形狀，每個Agents的觀察空間都是list中的一個元素
    action_n = 8 # 可透過print(env.action_space)得知Agent在此場景的動作空間為Discrete(8)
    action_shape_n = [] # 設定Agents動作空間的形狀
    for i in range(0, agent_n):
        obs_shape_n.append(obs_shape)
        action_shape_n.append(action_n)
        
    maddpg = get_trainers(obs_shape_n, action_shape_n) # 創立MADDPG架構的實例
    
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
    
    
    for episode in range(0, 10000):
        for step in range(0, 2000):
            print(step)
            # 以機率形式輸出Agents的動作
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            
            action_queue.put(action_n[1])
            event_act.set() # 等待action_n計算完成，再通知子程序
            
            # 返回的觀察資訊、獎勵、結束訊號、除錯資訊皆為list型態
            # 每個Agent為list中的一個元素
            new_obs_n, rew_n, done_n, info_n = env.step(action_n[0])
            # 將list轉換為ndarray以執行降維
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1) # 將三維的觀察資料降成一維
            new_obs_list = []
            new_obs_list.append(new_obs_n)
            
            
            event_obs.wait() # 等待子程序傳遞資料
            #lock.acquire()
            p1_obs = info_queue.get() # 取得子程序資料
            p1_rew = info_queue.get()
            p1_done = info_queue.get()
            #lock.release()
            
            # 測試用
            '''
            print('p1_obs:')
            print(p1_obs)
            print('p1_rew:')
            print(p1_rew)
            print('p1_done:')
            print(p1_done)
            '''
            new_obs_list.append(p1_obs)
            rew_n.append(p1_rew)
            done_n.append(p1_done)
            event_obs.clear()
            
            
            # 將資訊紀錄於MADDPG結構的記憶體
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_list, done_n)
            # 將所有Agents的章節獎勵加總
            episode_rewards[-1] += np.sum(rew_n)
            
            # 每個Agent之觀察都是list obs_n中的一個元素
            obs_n = []
            obs_n.append(new_obs_n)
            obs_n.append(new_obs_list[-1])
            
            done = all(done_n)
            #lock.acquire()
            info_queue.put(done) # 將結束旗標傳遞給子程序 
            #lock.release()
            event_done.set() # 等待存取完畢，再通知子程序
            
            if step % 400 == 0:
                # 更新神經網路，目前仍在修正中，尚未完成
                x = maddpg.memory.sample(6400)
                
                # 測試用
                '''
                print('this is x:')
                print(type(x))
                print(x)
                '''
                maddpg.update(x)
                maddpg.update_all_agents()
                
            # 檢查章節是否結束
            if done or step == 1999:
                # 每個Agent之觀察都是list obs_n中的一個元素
                # 但目前該場景只有一個Agent，故暫時以此方式賦值
                obs_n = []
                obs_tmp = env.reset()
                obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
                obs_n.append(obs_tmp)
                
                event_obs.wait() # 等待player1取得觀察資料後，再執行串接
                obs_n.append(info_queue.get())
                event_obs.clear() # 將訊號重置
                
                episode_rewards[-1] = episode_rewards[-1]/step
                print(episode_rewards[-1])
                episode_rewards.append(0.0) # 初始化下一章節的獎勵
                break
            
        # 每隔100章節儲存一次訓練模型
        if episode % 100 == 0:
            maddpg.save_model(episode)



if __name__ == '__main__':
    
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
    
    player1_proc = Process(target=player1, args=(info_queue, action_queue, lock, event_obs, event_act, event_done))
    player1_proc.start()
    train()
    player1_proc.join()
    #play()