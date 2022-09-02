#!/usr/bin/env python3

import os
from random import choice
from time import time
import vizdoom as vzd
from multiprocessing import Process
from threading import Thread

# Run this many episodes
episodes = 1

# maddpg_duel之場景內容為兩位玩家使用火箭發射器互相攻擊
# 當任意玩家死亡時，即結束一個章節
config = os.path.join(vzd.scenarios_path, "maddpg_duel.cfg")

def player1():
    game = vzd.DoomGame()
    game.load_config(config)
    game.add_game_args("-host 2 -deathmatch -netmode 0 +timelimit 1 +sv_spawnfarthest 1")
    game.add_game_args("+name Player1 +colorset 0")
    game.set_window_visible(True)
    # 放大畫面以方便觀察
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.init()
    
    if game.is_player_dead():
        game.respawn_player()
    
    # 動作包含往左、往右、攻擊
    actions = [[True, False, False], [False, True, False], [False, False, True]]

    for i in range(episodes):

        print("Episode #" + str(i + 1))

        while not game.is_episode_finished():
            #if game.is_player_dead():
                #game.respawn_player()

            game.make_action(choice(actions))

        print("Episode finished!")
        print("Player1 frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

        # Starts a new episode. All players have to call new_episode() in multiplayer mode.
        game.new_episode()

    game.close()


def player2():
    game = vzd.DoomGame()
    game.load_config(config)
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.add_game_args("-join 127.0.0.1")
    game.add_game_args("+name Player2 +colorset 3")
    game.init()

    # 動作包含往左、往右、攻擊
    actions = [[True, False, False], [False, True, False], [False, False, True]]
    
    if game.is_player_dead():
        game.respawn_player()
        
    for i in range(episodes):
        r = 0
        while not game.is_episode_finished():
            #if game.is_player_dead():
                #game.respawn_player()

            r_tmp = game.make_action(choice(actions))
            r = r + r_tmp
            print("reward: ", r)

        print("Player2 frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        game.new_episode()

    game.close()




if __name__ == '__main__':
    # Both Processes or Threads can be used to have many DoomGame instances running in parallel.
    # Because ViZDoom releases GIL, there is no/minimal difference in performance between Processes and Threads.
    start = time()
    #p1 = Process(target=player1)
    
    # 使用Thread模組較Process模組方便觀察
    p1 = Thread(target=player1)
    p1.start()
    player2()

    print("Finished", episodes, "episodes after", time() - start, "seconds")
