import numpy as np
import os


class Predictor_15(object):
    
    def __init__(self):
        
        # load model
        lll = np.load(os.path.dirname(__file__)+"/data/vilside_model_71.npz")
        
        self.W1_np = lll['arr_0']
        self.b1_np = lll['arr_1']
        self.W2_np = lll['arr_2']
        self.b2_np = lll['arr_3']
        
        # num of param
        self.n_para_3d = 5
        self.n_para_2d = 5
        
        self.x_3d = np.zeros((15, 15, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((15, self.n_para_2d), dtype='float32')
        
                
    def initialize(self, base_info, game_setting):
        # game_setting
        self.game_setting = game_setting
        
        # base_info
        self.base_info = base_info
        
        # initialize x_3d, x_2d
        self.x_3d = np.zeros((15, 15, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((15, self.n_para_2d), dtype='float32')
        
        
        """
        X_3d
        [i, j, 0] : agent i voted agent j (not in talk, action)
        [i, j, 1] : agent i divined agent j HUMAN
        [i, j, 2] : agent i divined agent j WEREWOLF
        [i, j, 3] : agent i inquested agent j HUMAN
        [i, j, 4] : agent i inquested agent j WEREWOLF
        # [i, j, 5] : agent i managed to guard agent j
        
        X_2d
        [i, 0] : agent i is executed
        [i, 1] : agent i is attacked
        [i, 2] : agent i comingout himself/herself SEER
        [i, 3] : agent i comingout himself/herself MEDIUM
        [i, 4] : agent i comingout himself/herself BODYGUARD
        # [i, 5] : agent i comingout himself/herself VILLAGER
        # [i, 6] : agent i comingout himself/herself POSSESSED
        # [i, 7] : agent i comingout himself/herself WEREWOLF
        """
        
        
    def update(self, gamedf):
        # read log
        for i in range(gamedf.shape[0]):
            # vote
            if gamedf.type[i] == 'vote' and gamedf.turn[i] == 0:
                self.x_3d[gamedf.idx[i] - 1, gamedf.agent[i] - 1, 0] += 1
            # execute
            elif gamedf.type[i] == 'execute':
                self.x_2d[gamedf.agent[i] - 1, 0] = 1
            # attacked
            elif gamedf.type[i] == 'dead':
                self.x_2d[gamedf.agent[i] - 1, 1] = 1
            # talk
            elif gamedf.type[i] == 'talk':
                content = gamedf.text[i].split()
                # comingout
                if content[0] == 'COMINGOUT':
                    # self
                    if int(content[1][6:8]) == gamedf.agent[i]:
                        if content[2] == 'SEER':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 2] = 1
                        elif content[2] == 'MEDIUM':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 3] = 1
                        elif content[2] == 'BODYGUARD':
                            self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                            self.x_2d[gamedf.agent[i] - 1, 4] = 1
                # divined
                elif content[0] == 'DIVINED':
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                    self.x_2d[gamedf.agent[i] - 1, 2] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 0
                # identified
                elif content[0] == 'IDENTIFIED':
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:5] = 0
                    self.x_2d[gamedf.agent[i] - 1, 3] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 3] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 4] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 4] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 3] = 0
        
    def ret_pred(self, conditions):
        x_np = np.zeros((15, 15, 20))
        # info
        x_np[:, :, 0:5] = self.x_3d
        for i in range(15):
            x_np[:, i,  5:10] = self.x_2d
            x_np[i, :, 10:15] = self.x_2d
        # known truth
        # conditions: [[agentid, role]]
        for c in conditions:
            k = c[0]
            role = c[1]
            role_num = 10
            if role == "WEREWOLF":
                role_num = 5
            elif role == "POSSESSED":
                role_num = 4
            elif role == "VILLAGER":
                role_num = 3
            elif role == "BODYGUARD":
                role_num = 2
            elif role == "MEDIUM":
                role_num = 1
            elif role == "SEER":
                role_num = 0
            # not valid for werewolves
            if role_num <= 5:
                x_np[k-1, :, 15+role_num] = 1
            
        # fit
        in1_np  = np.tensordot(x_np, self.W1_np[0,0,:,:], axes=[[2], [0]]) + self.b1_np
        out1_np = 1.0 / (1.0 + np.exp(-in1_np))
        x2_np = out1_np.mean(axis=1)
        in2_np  = np.tensordot(x2_np, self.W2_np[0,0,:,:], axes=[[1], [0]]) + self.b2_np
        y_np = np.exp(in2_np)
        y_np = y_np / (np.matmul(y_np.sum(axis=1).reshape((15, 1)), np.ones((1, 6))))
        
        return y_np


        