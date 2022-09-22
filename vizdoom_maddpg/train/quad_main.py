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

# 創立MADDPG架構的實例
def get_trainers(modelname,agent_num, obs_shape_n, action_shape_n):
    return MADDPG(modelname,agent_num, obs_shape_n, action_shape_n, 0.7, 20000)

def player(host_arg, agent_arg, player_queue, lock, event_obs, event_act, event_done):
    env = gym.make('VizdoomMultipleInstances-v0', host=host_arg, agent_num=agent_arg) # host參數為0意指創建本地伺服器端
    obs_tmp = env.reset()
    obs_tmp = obs_tmp.reshape(-1)
    player_queue.put(obs_tmp)
    event_obs.set() # 存取首次觀察資料並傳遞後，通知主程序
    
    for episode in range(0, 10000):
        for step in range(0, 2000):
            # 當場景創建時，場內角色會死亡，因此必須先將其復活
            env.check_is_player_dead()
            event_act.wait() # 等待主程序傳遞動作資料
            new_obs_n, rew_n, done_n, info_n = env.step(player_queue.get())
            event_act.clear() # 重置訊號
              
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1)
            # 將新的觀察資料、獎勵、完成訊號傳入主程序
            player_queue.put(new_obs_n)
            player_queue.put(rew_n[-1])
            player_queue.put(done_n[-1])
            event_obs.set() # 採取動作並完成處理後，將資料傳遞給主程序
            
            event_done.wait() # 等待父程序傳遞結束旗標
            done = player_queue.get() # 取得結束旗標
            event_done.clear()
            
            if done or step == 1999:
                obs_tmp = env.reset()
                obs_tmp = obs_tmp.reshape(-1)
                player_queue.put(obs_tmp)
                event_obs.set() # 存取觀察資料並傳遞後，通知主程序
                break
                
            
    
# 主要訓練函式
def train():
    env = gym.make('VizdoomMultipleInstances-v0', host=3) # host參數不為0意指加入本地伺服器的客戶端
    
    obs_shape = 120*160*3 # 觀察空間為的高為120、寬為160、頻道數為3(RGB)
    obs_shape_n = []  # 設定Agents觀察空間的形狀，每個Agents的觀察空間都是list中的一個元素
    action_n = 8 # 可透過print(env.action_space)得知Agent在此場景的動作空間為Discrete(8)
    action_shape_n = [] # 設定Agents動作空間的形狀
    for i in range(0, agent_num):
        obs_shape_n.append(obs_shape)
        action_shape_n.append(action_n)
        
    maddpg = get_trainers('MaddpgQuad',agent_num, obs_shape_n, action_shape_n) # 創立MADDPG架構的實例
    # 初始化章節獎勵
    episode_rewards = [0.0] 
    # 每個Agent之觀察都是list obs_n中的一個元素
    obs_n = []
    obs_tmp = env.reset() # env.reset()初始化環境，並返回Agent初始觀察資料
    obs_tmp = obs_tmp.reshape(-1) # 將三維的觀察資料降成一維
    obs_n.append(obs_tmp)
    
    # 無法使用迴圈，因執行較快的子程序可能略過其訊號等待的過程
    event_obs_list[1].wait() # 等待player1取得首次觀察資料後在執行串接
    obs_n.append(queue_list[1].get())
    event_obs_list[2].wait() # 等待player2取得首次觀察資料後在執行串接
    obs_n.append(queue_list[2].get())
    event_obs_list[3].wait() # 等待player3取得首次觀察資料後在執行串接
    obs_n.append(queue_list[3].get())
    # 通知各子程序可繼續執行
    for player_no in range(1, agent_num):
        event_obs_list[player_no].clear()
    
    
    for episode in range(0, 10000):
        for step in range(0, 2000):
            # 當場景創建時，場內角色會死亡，因此必須先將其復活
            env.check_is_player_dead()
            # print(step)
            # 以機率形式輸出Agents的動作
            action_n = [agent.act_prob(torch.from_numpy(obs.astype(np.float32)).to(device)).detach().cpu().numpy()
                        for agent, obs in zip(maddpg.agents, obs_n)]
            
            # 取得機率值最大的動作索引值，並轉換為整數資料型態
            p0_action = int(np.argmax(action_n[0]))
            for agent_no in range(1, agent_num):
                queue_list[agent_no].put(int(np.argmax(action_n[agent_no])))
                event_act_list[agent_no].set() # 等待action_n計算完成，再通知子程序
            
            # 返回的觀察資訊、獎勵、結束訊號、除錯資訊皆為list型態
            # 每個Agent為list中的一個元素
            new_obs_n, rew_n, done_n, info_n = env.step(p0_action)
            # 將list轉換為ndarray以執行降維
            new_obs_n = np.array(new_obs_n)
            new_obs_n = new_obs_n.reshape(-1) # 將三維的觀察資料降成一維
            new_obs_list = []
            new_obs_list.append(new_obs_n)
            
            
            # 無法使用迴圈，因執行較快的子程序可能略過其訊號等待的過程
            event_obs_list[1].wait() # 等待子程序傳遞資料
            new_obs_list.append(queue_list[1].get()) # 取得子程序之新觀察資料
            rew_n.append(queue_list[1].get()) # 取得子程序之獎勵
            done_n.append(queue_list[1].get()) # 取得子程序之結束訊號
            event_obs_list[2].wait()
            new_obs_list.append(queue_list[2].get())
            rew_n.append(queue_list[2].get())
            done_n.append(queue_list[2].get())
            event_obs_list[3].wait()
            new_obs_list.append(queue_list[3].get())
            rew_n.append(queue_list[3].get())
            done_n.append(queue_list[3].get())
            # 通知各子程序可繼續執行
            for player_no in range(1, agent_num):
                event_obs_list[player_no].clear()
            
            
            # 將資訊紀錄於MADDPG結構的記憶體
            maddpg.add_data(obs_n, action_n, rew_n, new_obs_list, done_n)
            # 將所有Agents的章節獎勵加總
            episode_rewards[-1] += np.sum(rew_n)
            
            # 每個Agent之觀察都是list obs_n中的一個元素
            obs_n = new_obs_list
            
            # 計算結束訊號
            done = all(done_n)
            
            for player_no in range(1, agent_num):
                queue_list[player_no].put(done) # 將結束旗標傳遞給子程序 
                event_done_list[player_no].set() # 等待存取完畢，再通知子程序
            
            if step % 400 == 0:
                # 更新神經網路，目前仍在修正中，尚未完成
                x = maddpg.memory.sample(batch_size)
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
                
                # 無法使用迴圈，因執行較快的子程序可能略過其訊號等待的過程
                event_obs_list[1].wait() # 等待子程序重置章節、取得觀察資料後，再執行串接
                obs_n.append(queue_list[1].get())
                event_obs_list[2].wait()
                obs_n.append(queue_list[2].get())
                event_obs_list[3].wait()
                obs_n.append(queue_list[3].get())
                
                # 通知各子程序可繼續執行
                for player_no in range(1, agent_num):
                    event_obs_list[player_no].clear()
                
                episode_rewards[-1] = episode_rewards[-1]/step
                print(episode_rewards[-1])
                episode_rewards.append(0.0) # 初始化下一章節的獎勵
                break
            
        # 每隔100章節儲存一次訓練模型
        if episode % 100 == 0:
            maddpg.save_model(episode)



if __name__ == '__main__':
    
    agent_num = 4 # 該場景的agents數量
    batch_size = 100
    
    manager = Manager()
    # 創建queue作為主程序與子程序的溝通管道
    queue_list = []
    # 創建lock以確保同時間只能有一個程序存取變數
    lock = manager.Lock()
    # 創建event以協助程序溝通
    event_obs_list = []
    event_act_list = []
    event_done_list = []
    for i in range(0, 4):
        queue_list.append(Queue())
        event_obs_list.append(Event())
        event_act_list.append(Event())
        event_done_list.append(Event())
    
    proc_list = []
    # 由於需要先建立本地伺服器，故第一個子程序的host參數必須為0
    proc_list.append(Process(target=player, args=(0, agent_num, queue_list[1], lock, event_obs_list[1], event_act_list[1], event_done_list[1])))
    proc_list.append(Process(target=player, args=(1, agent_num, queue_list[2], lock, event_obs_list[2], event_act_list[2], event_done_list[2])))
    proc_list.append(Process(target=player, args=(2, agent_num, queue_list[3], lock, event_obs_list[3], event_act_list[3], event_done_list[3])))
    
    for proc in proc_list:
        proc.start()
    train()
    for proc in proc_list:
        proc.join()
    #play()