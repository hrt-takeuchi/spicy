#!/usr/bin/env python
from __future__ import print_function, division 

# this is main script

import aiwolfpy
import aiwolfpy.contentbuilder as cb
import numpy as np

#chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, optimizers, Variable
from chainer import serializers

# sample 
import aiwolfpy.spicy
import random
from savelog import save_log

# 時間計測用
import time
import os
# 結果出力
from read_record import read_record,save_file_at_new_dir


from aiwolfpy.forCal import dead_or_alive 

## 正規表現 0718
import re



myname = "Spicy"

class SpicyPlayer(object):
    

    def __init__(self, agent_name):
        # myname
        self.myname = agent_name

        # 勝数、陣営勝数の定義
        self.win_rate = [0 for i in range(15)]
        self.were_win_rate = [0 for i in range(15)]
        self.vila_win_rate = [0 for i in range(15)]

        
        
    def getName(self):
        return self.myname
    


    def initialize(self, base_info, diff_data, game_setting):
        # print(base_info)
        # print(diff_data)
        # base_info
        self.base_info = base_info
        # game_setting
        self.game_setting = game_setting
        self.playerNum = self.game_setting['playerNum']
       
        # initialize
        if self.game_setting['playerNum'] == 15:
             # 人間リスト
            self.humList = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
            # カミングアウトリスト
            self.comingout_list = [0 for i in range(15)]
            #特徴量
            self.feature_value = np.zeros((15, 11), dtype='float32') #0~14
            #推定スコアリスト
            self.estimate_score = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0, '7': 0, '8': 0, '9': 0, '10': 0, '11': 0, '12': 0, '13': 0, '14': 0, '15': 0}
            #モデルファイル名
            model_file = "/aiwolfpy/spicy/data/mymodel0718.h5"
        elif self.game_setting['playerNum'] == 5:
             # 人間リスト
            self.humList = [0,1,2,3,4]
            # カミングアウトリスト
            self.comingout_list = [0 for i in range(5)]
            if len(self.win_rate) == 15:
                del self.win_rate[4:14]
                del self.vila_win_rate[4:14]
                del self.were_win_rate[4:14]
            #特徴量
            self.feature_value = np.zeros((5, 11), dtype='float32') #0~14
            #推定スコアリスト
            self.estimate_score = {'1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            #モデルデータ名
            model_file = "/aiwolfpy/spicy/data/mymodel0718.h5"

        # 狼確定リスト
        self.wolfList = []
        ################################################################################
        # 自分が人狼の時の狼リスト     
        if self.base_info['myRole'] == 'WEREWOLF':
            for i in range(diff_data.shape[0]):
                    if diff_data['type'][i] == 'initialize' and diff_data['text'][i].split()[2] == 'WEREWOLF':
                        self.wolfList.append(diff_data['idx'][i])
            # print("狼リスト")
            # print(self.wolfList)
        # 人狼の狂人確定リストとattackvote用カウンター
        self.possessedList = [] 
        self.attackSeerCount = 0
        self.realSeerNum = 0

        # 自分のID
        self.id = self.base_info['agentIdx']
                
        ### EDIT FROM HERE ###     
        self.divined_list = []
        self.comingout = ''
        self.myresult = ''
        self.not_reported = False
        self.vote_declare = 0
        self.pp_comingout = 0

        # # 各プレイヤーの発言数
        self.talk_number = [0 for i in range(15)]

        # カミングアウトリスト 0~14
        self.seeList = [] #占い師
        self.medList = [] #霊媒師
        self.posList = [] #狂人
        self.werList = [] #人狼
        self.seerBlackList = [] # 自分が占い師の時の対抗者のリスト
        self.seer_roller = 0
        # 投票数リスト
        self.vote_list = []
        # 生存者リスト
        self.aliveList = []
        # 村人陣営用ブラックリスト(黒出ししてきたら黒)
        self.blackList = []
 

        #変更１ model読み込み
        class DNN(Chain):
            def __init__(self):
                super(DNN, self).__init__(
                    l1 = L.Linear(11, 100),
                    l2 = L.Linear(100, 100),
                    l3 = L.Linear(100, 2)
                )
            def forward(self, x):
                h1 = F.relu(self.l1(x))
                h2 = F.relu(self.l2(h1))
                h3 = self.l3(h2)
                return h3
        self.model = DNN()
        file_path = os.path.dirname(__file__) + model_file
        self.model.to_cpu()
        serializers.load_hdf5(file_path, self.model)

        self.jinro_score = []


    def update(self, base_info, diff_data, request):
        # print(base_info)
        # print(diff_data)
        # print(self.win_rate)
        # update base_info
        self.base_info = base_info
        self.identifyResult = 0
        # 生存者リスト
        self.aliveList = [i for i, x in self.base_info['statusMap'].items() if x == 'ALIVE']
        # ランダム値の生成
        self.rand_rate = random.random()  

        # 各プレイヤーの発言数
        for i in range(diff_data.shape[0]):
            if diff_data['text'][i] != 'skip' and diff_data['text'][i] != 'over':
                self.talk_number[int(diff_data['agent'][i])-1] += 1

        # talk数で降順にidを並べたリスト
        self.talk_num_order_list = np.argsort(self.talk_number)[::-1] + 1

        # 黒出し霊媒用
        if base_info['myRole'] == 'MEDIUM':
            for i in range(diff_data.shape[0]):
                    if diff_data['type'][i] == 'identify' and diff_data['text'][i].split()[2] == 'WEREWOLF':
                        self.identifyResult = 1
                        self.myresult = diff_data['text'][i]


        #########################################################################
        # 占い師用ブラックリスト
        if base_info['myRole'] == 'SEER':
            for i in range(diff_data.shape[0]):
                if diff_data['type'][i] == 'divine' and diff_data['text'][i].split()[2] == 'WEREWOLF':
                    black = diff_data['text'][i].split()[1]
                    blackid = int(black.replace('Agent[','').replace(']',''))
                    self.wolfList.append(blackid)

        # 村人陣営用ブラックリスト(黒出ししてきたら黒)
        if base_info['myRole'] !=  'WEREWOLF' and base_info['myRole'] !=  'POSSESSED':
            for i in range(diff_data.shape[0]):
                if diff_data['type'][i] == 'talk' and diff_data['text'][i].split()[0] == 'DIVINED' and diff_data['text'][i].split()[2] == 'WEREWOLF' and diff_data['text'][i].split()[1] == ('Agent[' + str(self.id) +']'):
                    self.blackList.append(diff_data['agent'][i])

        # 投票あわせのためVOTEリスト 
        for i in range(diff_data.shape[0]):
            if diff_data['type'][i] == 'vote' and diff_data['text'][i].split()[0] == 'VOTE':
                vote_num = int(diff_data['text'][i].split()[1].replace('Agent[','').replace(']','')) - 1
                self.vote_list[vote_num] += 1

        # 人狼用の狂人確定リスト（人狼でない占いの自分への白だし）
        if base_info['myRole'] == 'WEREWOLF':
            for i in range(diff_data.shape[0]):
                if diff_data['type'][i] == 'talk' and diff_data['text'][i].split()[0] == 'DIVINED' and diff_data['text'][i].split()[2] != 'WEREWOLF' and diff_data['text'][i].split()[1] == ('Agent[' + str(self.id) +']'):
                    pos_num = int(diff_data['text'][i].split()[1].replace('Agent[','').replace(']',''))
                    if pos_num not in self.wolfList:
                        self.possessedList.append(pos_num)
            # 占い確定
        if len(self.seeList) > 1 and len(self.possessedList) > 0:
            # 占いCOリストから人狼リストと狂確リストを除外
            a = self.seeList
            b = self.possessedList
            c = self.wolfList
            set_abc = set(a) - set(b) - set(c)
            realSeer = list(set_abc)
            self.realSeerNum = realSeer[0]
        ############################################################################
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
        # カミングアウトリスト
        # 0:村人 1:占い師 2:霊媒師 3:狂人 4:人狼
        for i in range(diff_data.shape[0]):
            coid = int(diff_data['agent'][i]) - 1
            if diff_data['text'][i].split()[0]== 'COMINGOUT':
                if diff_data['text'][i].split()[2] == 'SEER':
                    self.comingout_list[coid] = 1
                    self.seeList.append(coid)
                    if coid in self.humList:
                        self.humList.remove(coid)
                    # ついでに自分が占い師ならブラックリスト登録
                    if self.base_info['myRole'] == 'SEER' and  self.id != coid:
                        self.seerBlackList.append(coid+1)
                elif (diff_data['text'][i].split()[2] == 'MEDIUM'):
                    self.comingout_list[coid] = 2
                    self.medList.append(coid)
                    if coid in self.humList:
                        self.humList.remove(coid)
                elif (diff_data['text'][i].split()[2] == 'POSSESSED'):
                    self.comingout_list[coid] = 3
                    self.posList.append(coid)
                    if coid in self.humList:
                        self.humList.remove(coid)
                elif (diff_data['text'][i].split()[2] == 'WEREWOLF'):
                    self.comingout_list[coid] = 4
                    self.werList.append(coid)
                    if coid in self.humList:
                        self.humList.remove(coid)
 
        # ローラー発動条件
        if len(self.seeList) == 3:
            self.seer_roller = 1

        #####　勝ち数計算 #######################################################################3
        # 1なら試合終了
        game_end = 0
        for i in range(diff_data.shape[0]):
            if diff_data['type'][i] == 'finish':
                game_end = 1
                break
        if game_end == 1:
            # 勝利陣営（0:村、1:狼）
            win_camp = 0
            wolf_camp = []
            if self.game_setting['playerNum'] == 15:
                for i in range(diff_data.shape[0]):
                    idnum = diff_data["agent"][i]
                    # 狼が生きてたら狼チームの勝利
                    if diff_data['type'][i] == 'finish' and diff_data['text'][i].split()[2] == 'WEREWOLF' or diff_data['text'][i].split()[2] == "POSSESSED":
                        wolf_camp.append(idnum)
                        if diff_data['text'][i].split()[2] == 'WEREWOLF' and self.base_info['statusMap'][str(idnum)] == 'ALIVE':
                            win_camp = 1       
                for i in range(1,16):
                    if win_camp == 1 and (i in wolf_camp):
                        self.were_win_rate[i-1] += 1
                        self.win_rate[i-1] += 1
                    elif win_camp == 0 and (i not in wolf_camp):
                        self.vila_win_rate[i-1] += 1
                        self.win_rate[i-1] += 1
            else: # 5人人狼
                for i in range(diff_data.shape[0]):
                    idnum = diff_data["agent"][i]
                    # 狼が生きてたら狼チームの勝利
                    if diff_data['type'][i] == 'finish' and diff_data['text'][i].split()[2] == 'WEREWOLF' or diff_data['text'][i].split()[2] == "POSSESSED":
                        wolf_camp.append(idnum)
                        if diff_data['text'][i].split()[2] == 'WEREWOLF' and self.base_info['statusMap'][str(idnum)] == 'ALIVE':
                            win_camp = 1       
                for i in range(1,6):
                    if win_camp == 1 and (i in wolf_camp):
                        self.were_win_rate[i-1] += 1
                        self.win_rate[i-1] += 1
                    elif win_camp == 0 and (i not in wolf_camp):
                        self.vila_win_rate[i-1] += 1
                        self.win_rate[i-1] += 1     
        #####　勝ち数計算 #######################################################################3

        for i in range(len(diff_data)):
            if 'talk' in diff_data['type'][i] and 'DIVINED' in diff_data['text'][i]:
                if diff_data['text'][i][18] =="H": #人間判定なら
                    #人間判定された回数「特徴量3」
                    self.feature_value[int(diff_data['text'][i][14:16])-1][2] += 1
                    #人間判定した回数「特徴量5」
                    self.feature_value[int(diff_data['agent'][i])-1][4] += 1
                elif diff_data['text'][i][18] =="W": #人狼判定なら
                    #人狼判定された回数「特徴量4」
                    self.feature_value[int(diff_data['text'][i][14:16])-1][3] += 1
                    #人狼判定した回数「特徴量6」
                    self.feature_value[int(diff_data['agent'][i])-1][5] += 1
                if diff_data['type'][i] == 'talk':
                    if 'ESTIMATE' in diff_data['text'][i]  and 'HUMAN' in diff_data['text'][i] or 'AGREE' in diff_data['text'][i]:
                        #賛成・信頼回数「特徴量8」
                        self.feature_value[int(diff_data['agent'][i])-1][8] += 1
                    if 'ESTIMATE' in diff_data['text'][i] and 'WEREWOLF' in diff_data['text'][i] or 'DISAGREE' in diff_data['text'][i]:
                        #反対・不信回数「特徴量」
                        self.feature_value[int(diff_data['agent'][i])-1][9] += 1

            #処刑なら1、襲撃なら2「特徴量7」
            if diff_data['type'][i] == 'execute':
                self.feature_value[int(diff_data['agent'][i])-1][7]=1
            elif diff_data['type'][i] == 'dead':
                self.feature_value[int(diff_data['agent'][i])-1][7]=2

        #推定スコア・更新
        y = self.model.forward(self.feature_value)
        for i in range(0, self.playerNum):
            self.estimate_score[str(i+1)] = float(str(y[i][1] - y[i][0]).replace("variable(","").replace(")",""))
        self.jinro_score = sorted(self.estimate_score.items(), key=lambda x: x[1])
   
    def dayStart(self):
        #日にち「特徴量1」
        for i in range(1, self.playerNum+1):
            self.feature_value[i-1][0] = self.feature_value[i-1][0] + 1
            #生死「特徴量2」
            if self.base_info['statusMap'][str(i)] == 'ALIVE':
                self.feature_value[i-1][6] = 1
            elif self.base_info['statusMap'][str(i)] == 'DEAD':
                self.feature_value[i-1][6] = 0
            #発言数「特徴量11」
            self.feature_value[i-1][10] = self.talk_number[i-1]

        self.vote_declare = 0
        self.talk_turn = 0
        # whisper用のID
        self.attackId = 0
        # 投票合わせ用のvoteリスト
        self.vote_list = [0 for i in range(15)]
        # リストから死亡削除 #########################################
        if len(self.humList) > 0:
            for num in self.humList:
                if self.base_info['statusMap'][str(num+1)] == 'DEAD':
                    if num in self.humList:
                        self.humList.remove(num)
        if len(self.seeList) > 0:
            for num in self.seeList:
                if self.base_info['statusMap'][str(num+1)] == 'DEAD':
                    self.seeList.remove(num)
        if len(self.medList) > 0:
            for num in self.medList:
                if self.base_info['statusMap'][str(num+1)] == 'DEAD':
                    self.medList.remove(num)
        if len(self.posList) > 0:
            for num in self.posList:
                if self.base_info['statusMap'][str(num+1)] == 'DEAD':
                    self.posList.remove(num)
        if len(self.werList) > 0:
            for num in self.werList:
                if self.base_info['statusMap'][str(num+1)] == 'DEAD':
                    self.werList.remove(num)
        if len(self.blackList) > 0:
            for num in self.blackList:
                if self.base_info['statusMap'][str(num)] == 'DEAD':
                    self.blackList.remove(num)
        if len(self.wolfList) > 0:
            for num in self.wolfList:
                if self.base_info['statusMap'][str(num)] == 'DEAD':
                    self.wolfList.remove(num)
        ###########################################################


        return None
    
    def talk(self):
        # パワープレイ
        if self.base_info['day'] > 1 and len(self.aliveList) == 3 and self.pp_comingout:
            if self.base_info['myRole'] == 'POSSESSED':
                self.comingout = 'POSSESSED'
                self.pp_comingout = False
                return cb.comingout(self.id, self.comingout)
            elif self.base_info['myRole'] == 'WEREWOLF':
                if len(self.seeList) == 2:
                    self.comingout = 'WEREWOLF'
                    self.pp_comingout = False
                    return cb.comingout(self.id, self.comingout)
            elif len(self.seeList) == 2 and len(self.werList) != 0:
                self.comingout = 'WEREWOLF'
                self.pp_comingout = False
                return cb.comingout(self.id, self.comingout)
        
        if self.game_setting['playerNum'] == 15:
            self.talk_turn += 1
            if self.base_info['myRole'] == 'SEER' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.id, self.comingout)
            # 黒発見でカミングアウト
            elif self.base_info['myRole'] == 'MEDIUM' and self.identifyResult == 1:
                self.not_reported = False
                self.comingout = 'MEDIUM'
                return cb.comingout(self.id, self.comingout), self.myresult
            # 狂った人
            elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
                if self.rand_rate > 0.15:
                    self.comingout = 'SEER'
                    return cb.comingout(self.id, self.comingout)
                # 霊媒師のふりするときも
                elif self.rand_rate > 0.05:
                    self.comingout = 'MEDIUM'
                    return cb.comingout(self.id, self.comingout)
                # たまに防人って言う
                elif self.rand_rate < 0.05:
                    self.comingout == 'BODYGUARD'
                    return cb.comingout(self.id, self.comingout)

            if self.base_info['myRole'] == 'SEER' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'MEDIUM' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
                self.not_reported = False
                idx = 1
                for i in range(1,15):
                    if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                        idx = int(self.jinro_score[-i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
                self.myresult = 'DIVINED Agent[' + "{0:02d}".format(idx) + '] ' + 'HUMAN'
                return self.myresult
            
            if self.vote_declare != self.vote():
                # 3日目に人狼は生存者リストからランダムに選んで投票先を発言する   
                if self.base_info['myRole'] == "WEREWOLF" and self.base_info['day'] == 3:
                    randlist = []
                    rand_num=len(self.aliveList)
                    rnd = int(random.uniform(1,rand_num))
                    self.vote_declare = int(self.aliveList[rnd])
                    return cb.vote(self.vote_declare)
                else:
                # 一番喋ってないやつに投票する
                    for i in range(0, 15):
                        idx_num = self.talk_num_order_list[i]
                        if self.base_info['statusMap'][str(idx_num)] == 'ALIVE' and idx_num != self.id:
                            self.vote_declare = idx_num
                            return cb.vote(self.vote_declare)
            self.vote_declare = self.vote()
            return cb.vote(self.vote_declare)

            if self.talk_turn <= 10:
                return cb.skip()

            return cb.over()
        # 5人人狼
        else:
            self.talk_turn += 1

            # 1.comingout anyway
            if self.base_info['myRole'] == 'SEER' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.id, self.comingout)
            elif self.base_info['myRole'] == 'POSSESSED' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.id, self.comingout)

            # 2.report
            if self.base_info['myRole'] == 'SEER' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'MEDIUM' and self.not_reported:
                self.not_reported = False
                return self.myresult
            elif self.base_info['myRole'] == 'POSSESSED' and self.not_reported:
                self.not_reported = False
                idx = 1
                for i in range(1,5):
                    if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                        idx = int(self.jinro_score[-i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
                self.myresult = 'DIVINED Agent[' + "{0:02d}".format(idx) + '] ' + 'HUMAN'
                return self.myresult



            # 3.declare vote if not yet
            if self.base_info['myRole'] != 'SEER' and self.talk_turn < 3:
                return cb.skip()
            elif self.base_info['myRole'] == 'POSSESSED' and self.base_info['day'] == 1:
                return cb.over()
            if self.vote_declare != self.vote():
                self.vote_declare = self.vote()
                return cb.vote(self.vote_declare)

            # 4. skip
            if self.talk_turn <= 10:
                return cb.skip()

            return cb.over()
   
    def whisper(self):
        if self.realSeerNum > 0:
            idx = self.realSeerNum
            return cb.estimate(idx, "SEER")
        if self.attackId == 0:
            attackId = self.attack()
            return cb.attack(attackId)
        return cb.skip()
        
    def vote(self):
        ## vote_listの投票多い順の並び替え
        voteList = np.argsort(self.vote_list)[::-1] + 1

        # パワープレイ 
        if len(self.aliveList) == 3 and self.base_info['day'] > 1:
            id_num = []
            if self.base_info['myRole'] == "WEREWOLF" or self.base_info['myRole'] == "POSSESSED":
                if len(self.humList) > 0:
                    id_num = self.humList
                elif len(self.werList) > 0:
                    if self.base_info['myRole'] == "WEREWOLF":
                        id_num = self.werList
                    elif self.base_info['myRole'] == "POSSESSED":
                        if len(self.seeList) > 0:
                            id_num = self.seeList
            if self.base_info['myRole'] == "HUMAN" or self.base_info['myRole'] == "SEER":
                if len(self.werList) > 0:
                    id_num = self.werList
                elif len(self.humList) > 0 and self.seer_roller == 0:
                    a = self.aliveList
                    b = self.seeList
                    set_abc = set(a) - set(b)
                    id_num = list(set_abc)
            if len(id_num) != 0:
                idx = int(id_num[0])
                return idx
        ## 15人人狼
        if self.game_setting['playerNum'] == 15:
            # 投票逃れ(自分のvoteがvote最大数と同じなら※最低2以上)
            if np.max(self.vote_list) == self.vote_list[self.id-1] and self.vote_list[self.id-1] > 1:
                for i in voteList:
                    if(i != self.id):
                        idx = int(i)
                        return idx
            if self.base_info['myRole'] == "WEREWOLF":
                idx = 1
                for i in range(1,15):
                    if self.base_info['statusMap'][self.jinro_score[i][0]] == 'ALIVE' and self.jinro_score[i][0] != str(self.id):
                            idx = int(self.jinro_score[i][0])
                    else:
                        continue
                    break
                return idx
            elif self.base_info['myRole'] == "POSSESSED":
                idx = 1
                for i in range(1,15):
                    if self.base_info['statusMap'][self.jinro_score[i][0]] == 'ALIVE' and self.jinro_score[i][0] != str(self.id):
                            idx = int(self.jinro_score[i][0])
                    else:
                        continue
                    break
                return idx

            # 黒出しへ投票
            if len(self.blackList) > 0:
                for i in self.blackList:
                    if i in voteList[0:2]:
                        idx = int(i)
                        return idx

            if self.base_info['myRole'] == "SEER":
                # 黒リストから投票
                if len(self.wolfList) > 0:
                    idx = 0
                    for i in self.wolfList:
                        if i in voteList[0:2]:
                            idx = int(i)
                    if idx == 0:
                        idx = int(self.wolfList[0])
                else: 
                    idx = 1
                    for i in range(1,15):
                        if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                            idx = int(self.jinro_score[-i][0])
                        else:
                            continue
                        break
                return idx
            # 占いローラー
            else:
                if self.seer_roller == 1 and len(self.seeList) > 0: #占い師ＣＯ者が3人以上なら
                    idx = 0
                    for i in self.seeList:
                        if i in voteList[0:2]:
                            idx = int(i)
                    if idx == 0:
                        idx = int(self.seeList[0])
                else:
                    for i in range(1,15):
                        if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                            idx = int(self.jinro_score[-i][0])
                        else:
                            continue
                        break
                return idx
           

            # else:
            #     # 一番喋ってないやつに投票する
            #     for i in range(0, 15):
            #         idx_num = self.talk_num_order_list[i]
            #         if self.base_info['statusMap'][str(idx_num)] == 'ALIVE':
            #             idx = idx_num
            #             break
            # return idx
        ## 5人人狼
        else:
            idx = 1
            if self.base_info['myRole'] == "WEREWOLF":
                for i in range(1,5):
                    if self.base_info['statusMap'][self.jinro_score[i][0]] == 'ALIVE' and self.jinro_score[i][0] != str(self.id):
                        idx = int(self.jinro_score[i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
            elif self.base_info['myRole'] == "POSSESSED":
                for i in range(1,5):
                    if self.base_info['statusMap'][self.jinro_score[i][0]] == 'ALIVE' and self.jinro_score[i][0] != str(self.id):
                        idx = int(self.jinro_score[i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
            # 村人側　初日は占いCO,自分以外に投票
            elif len(self.seeList) == 2 and self.base_info['day'] == 1:
                a = [1,2,3,4,5]
                b = self.seeList
                c =[self.id]
                set_abc = set(a) - set(b) - set(c)
                suspision = list(set_abc)
                ranum = len(suspision) - 1
                rnd = int(random.uniform(0,ranum))
                idx = int(suspision[rnd])

            elif self.base_info['myRole'] == "SEER":
                if len(self.wolfList) > 0:
                    idx = 0
                    for i in self.wolfList:
                        if i in voteList[0:2]:
                            idx = int(i)
                    if idx == 0:
                        idx = int(self.wolfList[0])

                for i in range(1,5):
                    if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                        idx = int(self.jinro_score[-i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
            else:
                for i in range(1,5):
                    if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                        idx = int(self.jinro_score[-i][0])
                        # print(self.jinro_score[-i][0] + 'に投票')
                    else:
                        continue
                    break
            return idx
    
    def attack(self):
        if self.game_setting['playerNum'] == 15:
            # 占い本確定１回噛み
            if self.realSeerNum > 0 and self.attackSeerCount == 0:
                idx = self.realSeerNum
                self.attackSeerCount = 1
                return idx

            # 強いやつから噛む
            ## vila_win_rateの勝ち数多い順の並び替え
            vilaWinList = np.argsort(self.vila_win_rate)[::-1] + 1
            for i in vilaWinList:
                if self.base_info['statusMap'][str(i)] == 'ALIVE':
                    idx = i
            return idx
        else:
            idx = 1
            for i in range(1,5):
                if self.base_info['statusMap'][self.jinro_score[i][0]] == 'ALIVE' and self.jinro_score[i][0] != str(self.id):
                    idx = int(self.jinro_score[i][0])
                    # print(self.jinro_score[-i][0] + 'に投票')
                else:
                    continue
                break
            return idx
    
    def divine(self):
        if self.game_setting['playerNum'] == 15:
            # 初日は勝率の高い人を占う
            if self.base_info['day'] == 0:
                self.strength = np.argsort(self.win_rate)
                idx = self.strength[0] + 1
            else:
                idx = 1
                for i in range(1,15):
                    if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                        idx = int(self.jinro_score[-i][0])
                    else:
                        continue
                    break
            return idx
        else:
            idx = 1
            for i in range(1,5):
                if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                    idx = int(self.jinro_score[-i][0])
                else:
                    continue
                break
            return idx
    
    def guard(self):
        if self.game_setting['playerNum'] == 15:
            idx = 1
            for i in range(1,15):
                if self.base_info['statusMap'][self.jinro_score[-i][0]] == 'ALIVE' and self.jinro_score[-i][0] != str(self.id):
                    idx = int(self.jinro_score[-i][0])
                    # print(self.jinro_score[-i][0] + 'に投票')
                else:
                    continue
                break
            return idx
        else:
            return 1
    
    def finish(self):
        pass
    

agent = SpicyPlayer(myname)

# run
if __name__ == '__main__':

    # top = '../log/'
    # for root, dirs, files in os.walk(top, topdown=False):
    #   for name in files:
    #       os.remove(os.path.join(root, name))
    #   for name in dirs:
    #       os.rmdir(os.path.join(root, name))
    # t1 = time.time()

    aiwolfpy.connect(agent)

    # t2 = time.time()
    # # 経過時間を表示
    # elapsed_time = str('{:.2f}'.format(t2-t1))
    # print(f"経過時間：{elapsed_time}")
    # # 大会時コメントアウト
    # read_record()
    # 