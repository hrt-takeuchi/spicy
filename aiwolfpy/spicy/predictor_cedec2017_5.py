from .tensor60 import Tensor60
import numpy as np
import os

class Predictor_5(object):
    
    def __init__(self):
        
        # load model
        lll = np.load(os.path.dirname(__file__)+"/data/model_60_0820.npz")
        
        self.W1_np = lll['arr_0']
        self.b1_np = lll['arr_1']
        self.W2_np = lll['arr_2']
        self.b2_np = lll['arr_3']
        self.W3_np = lll['arr_4']
        self.b3_np = lll['arr_5']
        
        # num of param
        self.n_para_3d = 12
        self.n_para_2d = 8
        
        self.x_3d = np.zeros((5, 5, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((5, self.n_para_2d), dtype='float32')
        
        self.case5 = Tensor60()
        self.t3d = self.case5.tensor60_3d
        self.t2d = self.case5.tensor60_2d
        
        self.t3d_mat = self.t3d.reshape(60, 5*5*4*4)
        self.t2d_mat = self.t2d.reshape(60, 5*4)
        
        self.watshi_xxx = np.ones((60, 4))
        
                
    def initialize(self, base_info, game_setting):
        # game_setting
        self.game_setting = game_setting
        
        # base_info
        self.base_info = base_info
        
        # initialize watashi_xxx
        self.watshi_xxx = np.ones((60, 4))
        xv = self.case5.get_case60_df()["agent_"+str(self.base_info['agentIdx'])].values
        self.watshi_xxx[xv != 0, 0] = 0.0
        self.watshi_xxx[xv != 1, 1] = 0.0
        self.watshi_xxx[xv != 2, 2] = 0.0
        self.watshi_xxx[xv != 3, 3] = 0.0
        
        # initialize x_3d, x_2d
        self.x_3d = np.zeros((5, 5, self.n_para_3d), dtype='float32')
        self.x_2d = np.zeros((5, self.n_para_2d), dtype='float32')
        
        
        """
        X_3d
        [i, j, 0] : agent i voted agent j (not in talk, action)
        [i, j, 1] : agent i divined agent j HUMAN
        [i, j, 2] : agent i divined agent j WEREWOLF
        
        X_2d
        [i, 0] : agent i is executed
        [i, 1] : agent i is attacked
        [i, 2] : agent i comingout himself/herself SEER
        [i, 3] : agent i comingout himself/herself MEDIUM
        [i, 4] : agent i comingout himself/herself BODYGUARD
        [i, 5] : agent i comingout himself/herself VILLAGER
        [i, 6] : agent i comingout himself/herself POSSESSED
        [i, 7] : agent i comingout himself/herself WEREWOLF
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
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 2] = 1
                        elif content[2] == 'MEDIUM':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 3] = 1
                        elif content[2] == 'BODYGUARD':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 4] = 1
                        elif content[2] == 'VILLAGER':
                            self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 5] = 1
                        elif content[2] == 'WEREWOLF':
                            self.x_2d[gamedf.agent[i] - 1, 7] = 0
                            self.x_2d[gamedf.agent[i] - 1, 6] = 1
                        elif content[2] == 'POSSESSED':
                            self.x_2d[gamedf.agent[i] - 1, 6] = 0
                            self.x_2d[gamedf.agent[i] - 1, 7] = 1
                # divined
                elif content[0] == 'DIVINED':
                    # 1, 2
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                    self.x_2d[gamedf.agent[i] - 1, 2] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 2] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 1] = 0
                elif content[0] == 'DIVINATION':
                    # 6
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                    self.x_2d[gamedf.agent[i] - 1, 2] = 1
                    # result
                    self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 6] = 1
                # identified
                elif content[0] == 'IDENTIFIED':
                    # 3, 4
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                    self.x_2d[gamedf.agent[i] - 1, 3] = 1
                    # result
                    if content[2] == 'HUMAN':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 3] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 4] = 0
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 4] = 1
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 3] = 0
                # guarded
                elif content[0] == 'GUARDED':
                    # 5
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                    self.x_2d[gamedf.agent[i] - 1, 4] = 1
                    # result
                    self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 5] = 1
                elif content[0] == 'GUARD':
                    # 7
                    # regard comingout
                    self.x_2d[gamedf.agent[i] - 1, 2:6] = 0
                    self.x_2d[gamedf.agent[i] - 1, 4] = 1
                    # result
                    self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 7] = 1
                # vote 
                elif content[0] == 'VOTE':
                    # 8
                    # keep recent
                    self.x_3d[gamedf.agent[i] - 1, :, 8] = 0
                    self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 8] = 1
                # estimate
                elif content[0] == 'ESTIMATE':
                    # 9-11
                    # keep recent
                    self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 9:12] = 0
                    if content[2] == 'POSSESSED':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 11] = 1
                    elif content[2] == 'WEREWOLF':
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 10] = 1
                    else:
                        self.x_3d[gamedf.agent[i] - 1, int(content[1][6:8])-1, 9] = 1
    
                                  
    def pred_60(self):
        
        x = np.concatenate([self.x_3d.reshape((1, 5*5*12)), self.x_2d.reshape((1, 5*8))], axis=1)
        
        u1 = np.matmul(x, self.W1_np) + self.b1_np
        z1 = np.minimum(np.maximum(u1, 0), 6)
        
        u2 = np.matmul(z1, self.W2_np) + self.b2_np
        z2 = np.minimum(np.maximum(u2, 0), 6)
        
        u3 = np.matmul(z2, self.W3_np) + self.b3_np
        u3 -= u3.max()
        z3 = np.exp(u3).reshape((60))
        
        return z3
        
    def ret_pred(self):
        p = self.pred_60()
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes = [0, 0]).transpose()
        
    def ret_pred_wx(self, r):
        p = self.pred_60() * self.watshi_xxx[:, r]
        return np.tensordot(self.case5.get_case60_2d(), p / p.sum(), axes = [0, 0]).transpose()


        