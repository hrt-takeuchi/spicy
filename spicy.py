#!/usr/bin/env python
from __future__ import print_function, division 
import random
# 時間計測用
import time
# 結果出力
from readrecord import read_record,save_file_at_new_dir

# this is main script

import aiwolfpy
import aiwolfpy.contentbuilder as cb

import shutil


# sample 
import aiwolfpy.spicy

import numpy as np
from aiwolfpy import savelog
# ログの削除用
import os


myname = 'spicy'

games_number = 0

class PythonPlayer(object):
    
    def __init__(self, agent_name):
        # myname
        self.myname = agent_name
        
        # predictor from sample
        # DataFrame -> P
        self.predicter_15 = aiwolfpy.spicy.Predictor_15()
        self.predicter_5 = aiwolfpy.spicy.Predictor_5()
        print(self.predicter_15)
        print(self.predicter_5)
        
    def getName(self):
        return self.myname
        
    def initialize(self, base_info, diff_data, game_setting):
        # print(base_info)
        # save_log("test", base_info)
        # print(diff_data)
        # base_info
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        
        # initialize
        if self.game_setting['playerNum'] == 15:
            self.predicter_15.initialize(base_info, game_setting)
        elif self.game_setting['playerNum'] == 5:   
            self.predicter_5.initialize(base_info, game_setting)
                
        ### EDIT FROM HERE ###     
        self.divined_list = []
        self.comingout = ''
        self.myresult = ''
        self.not_reported = False
        self.vote_declare = 0
        self.vote_15 = np.zeros((15, 15))
        self.vote_5 = np.zeros((5, 5))

        
    def update(self, base_info, diff_data, request):
        # print(base_info['statusMap'])
        # print(base_info)
        # print(diff_data)
        # update base_info
        self.base_info = base_info
        
        # result
        if request == 'DAILY_INITIALIZE':
            for i in range(diff_data.shape[0]):
                # IDENTIFY
                if diff_data['type'][i] == 'identify':
                    self.not_reported = True
                    self.myresult = diff_data['text'][i]
                    
                # DIVINE
                if diff_data['type'][i] == 'divine':
                    self.not_reported = True
                    self.myresult = diff_data['text'][i]
                    
                # GUARD
                if diff_data['type'][i] == 'guard':
                    self.myresult = diff_data['text'][i]
                
            # POSSESSED
            if self.base_info['myRole'] == 'POSSESSED':
                self.not_reported = True
                
        # UPDATE
        if self.game_setting['playerNum'] == 15:
            # update pred
            self.predicter_15.update(diff_data)
        else:
            self.predicter_5.update(diff_data)
            
        
        
    def dayStart(self):
        self.vote_declare = 0
        self.talk_turn = 0
        self.whis_declare = 0
        day_num = str(self.base_info['day'])
        global games_number
        if day_num == "0":
            games_number += 1
            print(str(games_number) + "試合目開始")
        print( day_num+"日目" ) 
        return None
    
    def talk(self):
        if self.game_setting['playerNum'] == 15:
            
            self.talk_turn += 1
            
            # 1.comingout anyway
            if self.base_info['myRole'] == 'SEER' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            elif self.base_info['myRole'] == 'MEDIUM' and self.comingout == '':
                self.comingout = 'MEDIUM'
                new_file_content = str(self.base_info)
             

                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)

            # 2.report
            if self.base_info['myRole'] == 'SEER' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'MEDIUM' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
                self.not_reported = False
                # FAKE DIVINE
                # highest prob ww in alive agents
                p = -1
                idx = 1
                p0_mat = self.predicter_15.ret_pred()
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
                self.myresult = 'DIVINED Agent[' + "{0:02d}".format(idx) + '] ' + 'HUMAN'
                return self.myresult
            # 3日目に人狼は生存者リストからランダムに選んでTalk先を決める   
            elif self.base_info['myRole'] == "WEREWOLF" and self.base_info['day'] == 3:
                randlist = []
                for i in range(1, 16):
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' :
                        randlist.append(i)
                rand_num=len(randlist)
                rnd = int(random.uniform(1,rand_num))
                print('hito ha:'+str(rnd))
                self.myresult = 'DIVINED Agent[' + "{0:02d}".format(rnd) + '] ' + 'HUMAN'
                return self.myresult
                
            # 3.declare vote if not yet
            if self.vote_declare != self.vote():
                self.vote_declare = self.vote()
                return cb.vote(self.vote_declare)
                
            # 4. skip
            if self.talk_turn <= 10:
                return cb.skip()
                
            return cb.over()
        else:
            self.talk_turn += 1
            
            # 1.comingout anyway
            if self.base_info['myRole'] == 'SEER' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            elif self.base_info['myRole'] == 'MEDIUM' and self.comingout == '':
                self.comingout = 'MEDIUM'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            
            # 2.report
            if self.base_info['myRole'] == 'SEER' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'MEDIUM' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
                self.not_reported = False
                # FAKE DIVINE
                # highest prob ww in alive agents
                p = -1
                idx = 1
                p0_mat = self.predicter_5.ret_pred_wx(2)
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
                self.myresult = 'DIVINED Agent[' + "{0:02d}".format(idx) + '] ' + 'HUMAN'
                return self.myresult
                
            # 3.declare vote if not yet
            if self.vote_declare != self.vote():
                self.vote_declare = self.vote()
                return cb.vote(self.vote_declare)
                
            # 4. skip
            if self.talk_turn <= 10:
                return cb.skip()
                
            return cb.over()
    
    def whisper(self):
        if self.whis_declare == 0:
            self.whis_declare += 1
            return cb.comingout(self.base_info['agentIdx'], 'VILLAGER')
        else:
            return cb.skip()
        
    def vote(self):
        
        if self.game_setting['playerNum'] == 15:
            # count vote
            self.vote_15 = self.predicter_15.x_3d[:, :, 8]
            for i in range(1, 16):
                if self.base_info['statusMap'][str(i)] == 'DEAD':
                    self.vote_15[i-1, :] = 0
            self.vote_15[self.base_info['agentIdx']-1, :] = 0
            p0_mat = self.predicter_15.ret_pred_wn()
            if self.base_info['myRole'] == "WEREWOLF":
                p = -1
                idx = 1
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if str(i) in self.base_info['roleMap'].keys():
                        p0 *= 0.5
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*1.0)
                        idx = i
            elif self.base_info['myRole'] == "POSSESSED":
                p = -1
                idx = 1
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*1.0)
                        idx = i
            else:
                # highest prob ww in alive agents provided watashi ningen
                p = -1
                idx = 1
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*0.5)
                        idx = i
            return idx
        else:
            self.vote_5 = self.predicter_5.x_3d[:, :, 8]
            if self.base_info['myRole'] == "WEREWOLF":
                p0_mat = self.predicter_5.ret_pred_wx(1)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 3]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*1.0)
                        idx = i
            elif self.base_info['myRole'] == "POSSESSED":
                p0_mat = self.predicter_5.ret_pred_wx(2)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 3]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*1.0)
                        idx = i
            elif self.base_info['myRole'] == "SEER":
                p0_mat = self.predicter_5.ret_pred_wx(3)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*0.5)
                        idx = i
            else:
                p0_mat = self.predicter_5.ret_pred_wx(0)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0 * (1.0 + self.vote_15[:, i-1].sum()*0.5)
                        idx = i
            return idx
    
    def attack(self):
        if self.game_setting['playerNum'] == 15:
            # highest prob hm in alive agents
            p = -1
            idx = 1
            p0_mat = self.predicter_15.ret_pred()
            for i in range(1, 16):
                p0 = p0_mat[i-1, 0]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p and i not in self.base_info['roleMap'].keys():
                    p = p0
                    idx = i
            return idx
        else:
            # lowest prob ps in alive agents
            p = 1
            idx = 1
            p0_mat = self.predicter_5.ret_pred_wx(1)
            for i in range(1, 6):
                p0 = p0_mat[i-1, 2]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 < p and i != self.base_info['agentIdx']:
                    p = p0
                    idx = i
            return idx
    
    def divine(self):
        if self.game_setting['playerNum'] == 15:
            # highest prob ww in alive and not divined agents provided watashi ningen
            p = -1
            idx = 1
            p0_mat = self.predicter_15.ret_pred_wn()
            for i in range(1, 16):
                p0 = p0_mat[i-1, 1]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and i not in self.divined_list and p0 > p:
                    p = p0
                    idx = i
            self.divined_list.append(idx)
            return idx
        else:
            # highest prob ww in alive and not divined agents provided watashi ningen
            p = -1
            idx = 1
            p0_mat = self.predicter_5.ret_pred_wx(3)
            for i in range(1, 6):
                p0 = p0_mat[i-1, 1]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and i not in self.divined_list and p0 > p:
                    p = p0
                    idx = i
            self.divined_list.append(idx)
            return idx
    
    def guard(self):
        if self.game_setting['playerNum'] == 15:
            # highest prob hm in alive agents
            p = -1
            idx = 1
            p0_mat = self.predicter_15.ret_pred()
            for i in range(1, 16):
                p0 = p0_mat[i-1, 0]
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
            return idx
        else:
            # no need
            return 1
    
    def finish(self):
        pass
        
 

agent = PythonPlayer(myname)

# run
if __name__ == '__main__':
    top = '../log/'
    for root, dirs, files in os.walk(top, topdown=False):
      for name in files:
          os.remove(os.path.join(root, name))
      for name in dirs:
          os.rmdir(os.path.join(root, name))
    t1 = time.time()
    aiwolfpy.connect_parse(agent)
    t2 = time.time()
    # 経過時間を表示
    elapsed_time = str('{:.2f}'.format(t2-t1))
    print(f"経過時間：{elapsed_time}")
    read_record()