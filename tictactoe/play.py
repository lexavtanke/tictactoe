import os
import time
import random
import io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from .env import TicTacToe
from .model import Policy


def play(model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    model = load_model(model_name + ".pt", device)
    
    done = False
    obs = env.reset()
    exp = {}
    
    player = 0 if random.random() > 0.5 else 1
    while not done:
        time.sleep(1)
        os.system("clear")
        print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
        
        env.render()
        
        action = None
        
        if player == 1:
            action = int(input())
        else:
            time.sleep(1)
            action = act(model, torch.tensor([obs], dtype=torch.float).to(device)).item()
            
        obs, reward, done, exp = env.step(action)
        player = 1 - player
    
    os.system("clear")
    print("Commands:\n{}|{}|{}\n-----\n{}|{}|{}\n-----\n{}|{}|{}\n\nBoard:".format(*[x for x in range(0, 9)]))
    env.render()
    print("reward ", reward, exp)
    if "tied" in exp["reason"]:
        print("A strange game. The only winning move is not to play.")
    exit(0)

def self_play(model1_name, model2_name, n_games=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TicTacToe()
    model1 = load_model(model1_name + ".pt", device)
    model2 = load_model(model2_name + ".pt", device)
    

    game_stat = {"player1" : 0,
                 "player2" : 0,
                 "ties": 0}
    
    for i in range(n_games):
        print(f"game number {i}")
        done = False
        obs = env.reset()
        exp = {}

        player = 0 if random.random() > 0.5 else 1
        while not done:
            time.sleep(1)
            # os.system("clear")
            print("Board:".format(*[x for x in range(0, 9)]))
            
            env.render()
            
            action = None
            
            if player == 1:
                action = act(model1, torch.tensor([obs], dtype=torch.float).to(device)).item()
            else:
                time.sleep(1)
                action = act(model2, torch.tensor([obs], dtype=torch.float).to(device)).item()
                
            obs, reward, done, exp = env.step(action)
            player = 1 - player
        
        # os.system("clear")
        print("Board:".format(*[x for x in range(0, 9)]))
        env.render()
        print("reward ", reward, exp)
        if "tied" in exp["reason"]:
            # print("A strange game. The only winning move is not to play.")
            game_stat["ties"] += 1
        elif "0 has won" in exp["reason"]:
            game_stat["player1"] +=1
        elif "1 has won" in exp["reason"]:
            game_stat["player2"] +=1

    
    print(f'after {n_games} game statistics is ',  game_stat)
    exit(0)


def load_model(path: str, device: torch.device):
    model = Policy(n_inputs=3*9, n_outputs=9).to(device)
    model_state_dict = torch.load(path, map_location=device)
    model.load_state_dict(model_state_dict)
    model.eval()
    return model

def act(model: Policy, state: torch.tensor):
    with torch.no_grad():
        p = F.softmax(model.forward(state)).cpu().numpy()
        valid_moves = (state.cpu().numpy().reshape(3,3,3).argmax(axis=2).reshape(-1) == 0)
        p = valid_moves*p
        return p.argmax()