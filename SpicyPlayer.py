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
myname = 'Spicy   '

from aiwolfpy.forCal import dead_or_alive

class SpicyPlayer(object):


    def __init__(self, agent_name):
        # myname
        self.myname = agent_name

        # predictor from sample
        # DataFrame -> P
        self.predicter_15 = aiwolfpy.spicy.Predictor_15()
        self.predicter_5 = aiwolfpy.spicy.Predictor_5()


        self.win_rate = [1 for i in range(15)]



    def getName(self):
        return self.myname



    def initialize(self, base_info, diff_data, game_setting):
        # print(base_info)
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
            if len(self.win_rate) == 15:
                del self.win_rate[4:14]





        ### EDIT FROM HERE ###
        self.divined_list = []
        self.comingout = ''
        self.myresult = ''
        self.not_reported = False
        self.vote_declare = 0

        # # 各プレイヤーの発言数
        self.talk_number = [0 for i in range(15)]

        # # 各プレイヤーの信頼度
        # self.confidence =  [50 for i in range(15)]

        # # 各プレイヤーのホワイトリスト
        # self.white_list = [0 for i in range(15)]

        # カミングアウトリスト
        self.comingout_list = [0 for i in range(15)]
        self.seeList = [] #占い師
        self.medList = [] #霊媒師
        self.posList = [] #狂人
        self.werList = [] #人狼
        self.seerBlackList = [] # 占い師ブラックリスト
        self.seer_roller = 0

        # 生存者リスト
        self.aliveList = []

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
        model = DNN()
        serializers.load_hdf5("mymodel0718.h5", model)

        #変更2 特徴量リスト--------------------------------------------------------
        self.feature_value = np.zeros((15, 11), dtype='float32')
        #-----------------------------------------------------------------------



    def update(self, base_info, diff_data, request):
        # print(base_info)
        #print(diff_data)
        #print('-----------------------------------------------------------------------------------------------------')
        # print(self.win_rate)
        # update base_info
        self.base_info = base_info
        self.identifyResult = 0

        #変更2 特徴量


        # 生存者リスト
        self.aliveList = [i for i, x in self.base_info['statusMap'].items() if x == 'ALIVE']
        # print(self.aliveList)


        # 各プレイヤーの発言数
        for i in range(diff_data.shape[0]):
            if diff_data['text'][i] != 'skip' and diff_data['text'][i] != 'over':
                self.talk_number[int(diff_data['agent'][i])-1] += 1


        self.talk_num_order_list = np.argsort(self.talk_number)[::-1] + 1

        # 黒出し霊媒用
        if base_info['myRole'] == 'MEDIUM':
            for i in range(diff_data.shape[0]):
                    if diff_data['type'][i] == 'identify' and diff_data['text'][i].split()[2] == 'WEREWOLF':
                        self.identifyResult = 1
                        self.myresult = diff_data['text'][i]


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

        # カミングアウトリスト
        # 0:村人 1:占い師 2:霊媒師 3:狂人 4:人狼
        for i in range(diff_data.shape[0]):
            coid = int(diff_data['agent'][i]) - 1
            if diff_data['text'][i].split()[0]== 'COMINGOUT':
                if diff_data['text'][i].split()[2] == 'SEER':
                    self.comingout_list[coid] = '1'
                    # ついでに自分が占い師ならブラックリスト登録
                    if self.base_info['myRole'] == 'SEER' and  self.base_info['agentIdx'] != coid:
                        self.seerBlackList.append(coid+1)

                elif (diff_data['text'][i].split()[2] == 'MEDIUM'):
                    self.comingout_list[coid] = '2'
                elif (diff_data['text'][i].split()[2] == 'POSSESSED'):
                    self.comingout_list[coid] = '3'
                elif (diff_data['text'][i].split()[2] == 'WEREWOLF'):
                    self.comingout_list[coid] = '4'

            self.seeList = [i for i, x in enumerate(self.comingout_list) if x == '1']
            self.medList = [i for i, x in enumerate(self.comingout_list) if x == '2']
            self.posList = [i for i, x in enumerate(self.comingout_list) if x == '3']
            self.werList = [i for i, x in enumerate(self.comingout_list) if x == '4']

        if len(self.aliveList) == 3:
            self.seer_roller = 1



    def dayStart(self):
        #変更-----------------------------------------------------------
        #日にち
        for i in range(0, 15):
            self.feature_value[i][0] = self.feature_value[i][0] + 1
            # if self.base_info['statusMap'][str(i)] == 'ALIVE':
            #     print("aaaaaaaa"
            #     self.feature_value[i][6] = 1
        print(self.feature_value)
        #生きていたら0,死んでいたら１
        print('---------------------------------------------')
        #------------------------------------------------------------

        self.vote_declare = 0
        self.talk_turn = 0
        return None

    def talk(self):
        rand_rate = random.random()
        if self.game_setting['playerNum'] == 15:

            self.talk_turn += 1

            # 1.comingout anyway

            if self.base_info['myRole'] == 'SEER' and self.comingout == '':
                self.comingout = 'SEER'
                return cb.comingout(self.base_info['agentIdx'], self.comingout)
            # 黒発見でカミングアウト
            elif self.base_info['myRole'] == 'MEDIUM' and self.comingout == '' and self.identifyResult == 1:
                self.not_reported = False
                self.comingout = 'MEDIUM'
                return cb.comingout(self.base_info['agentIdx'], self.comingout), self.myresult
            # 狂った人
            elif self.base_info['myRole'] == 'POSSESSED':
                # パワープレイ用
                if len(self.aliveList) == 3:
                    self.comingout = 'POSSESSED'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
                if self.comingout == '':
                        if rand_rate > 0.15:
                            self.comingout = 'SEER'
                            rand_rate = random.random()
                            return cb.comingout(self.base_info['agentIdx'], self.comingout)
                        # 霊媒師のふりするときも
                        elif rand_rate > 0.05:
                            self.comingout = 'MEDIUM'
                            rand_rate = random.random()
                            return cb.comingout(self.base_info['agentIdx'], self.comingout)
                        # たまに人狼って言う
                        elif rand_rate < 0.05:
                            self.comingout == 'WEREWOLF'
                            rand_rate = random.random()
                            return cb.comingout(self.base_info['agentIdx'], self.comingout)
                # # さらにカミングアウト
                # elif self.comingout == 'MEDIUM' and rand_rate > 0.7:
                #     self.comingout = 'SEER'
                #     return cb.comingout(self.base_info['agentIdx'], self.comingout)
                # elif self.comingout == 'SEER' and rand_rate > 0.9:
                #     self.comingout = 'MEDIUM'
                #     return cb.comingout(self.base_info['agentIdx'], self.comingout)
                # elif self.comingout == 'WEREWOLF' and rand_rate > 0.5:
                #     self.comingout = 'HUMAN'
                #     return cb.comingout(self.base_info['agentIdx'], self.comingout)
            # 村人の時
            elif self.base_info['myRole'] == 'VILLAGER' and self.comingout == '':
                # PP対策
                if len(self.aliveList) == 3:
                    self.comingout = 'WEREWOLF'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)

            # 人狼パワープレイ用
            elif self.base_info['myRole'] == "WEREWOLF":
            # ３人
                if len(self.aliveList) == 3 and len(dead_or_alive(self.posList,self.aliveList)) != 0:
                    self.comingout = 'WEREWOLF'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
            # ５人
                elif len(self.aliveList) == 5 and len(dead_or_alive(self.posList, self.aliveList)) != 0 and len(self.werList) == 2:
                    self.comingout = 'WEREWOLF'
                    return cb.comingout(self.base_info['agentIdx'], self.comingout)
            # # 7人
            #     elif len(self.aliveList) == 7 and len(dead_or_alive(self.posList, self.aliveList)) != 0 and len(self.werList) == 3:
            #         self.comingout = 'WEREWOLF'
            #         return cb.comingout(self.base_info['agentIdx'], self.comingout)



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


            # 3.declare vote if not yet
            if self.vote_declare != self.vote():
                # 3日目に人狼は生存者リストからランダムに選んで投票先を発言する
                if self.base_info['myRole'] == "WEREWOLF" and self.base_info['day'] == 3:
                    randlist = []
                    for i in range(1, 16):
                        if self.base_info['statusMap'][str(i)] == 'ALIVE' :
                            randlist.append(i)
                    rand_num=len(randlist)
                    rnd = int(random.uniform(1,rand_num))
                    return cb.estimate(rnd, "WEREWOLF")
                elif self.base_info['myRole'] == 'VILLAGER':
                # 一番喋ってないやつに投票する
                    for i in range(0, 15):
                        idx_num = self.talk_num_order_list[i]
                        if self.base_info['statusMap'][str(idx_num)] == 'ALIVE' and idx_num != self.base_info['agentIdx']:
                            return cb.estimate(idx_num, "WEREWOLF")
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
        return cb.skip()

    def vote(self):
        #変更-------------------------------------------------------------------
        for i in range(0, 15):
            self.feature_value[i][1] = len(self.seeList)
        #------------------------------------------------------------------------
        # パワープレイ
        if len(self.aliveList) == 3:
            id_num = []
            if self.base_info['myRole'] == "WEREWOLF" or self.base_info['myRole'] == "POSSESSED":
                id_num = [i for i, x in enumerate(self.comingout_list) if x == 0]
            if self.base_info['myRole'] == "HUMAN":
                id_num = dead_or_alive(self.werList,self.aliveList)

            if len(id_num) != 0:
                idx = id_num[0]
                return idx

        if self.game_setting['playerNum'] == 15:
            p0_mat = self.predicter_15.ret_pred_wn()
            if self.base_info['myRole'] == "WEREWOLF":
                p = -1
                idx = 1
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if str(i) in self.base_info['roleMap'].keys():
                        p0 *= 0.5
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
            elif self.base_info['myRole'] == "POSSESSED":
                p = -1
                idx = 1
                for i in range(1, 16):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
            elif self.base_info['myRole'] == "SEER":
                # 黒リストから投票
                if len(self.seerBlackList) > 0:
                    idx = 1
                    for i in self.seerBlackList:
                        if self.base_info['statusMap'][str(i)] == 'ALIVE':
                            idx = i
                    # print('黒吊りだぜ')
                else:
                    # highest prob ww in alive agents provided watashi ningen
                    p = -1
                    idx = 1
                    for i in range(1, 16):
                        p0 = p0_mat[i-1, 1]
                        if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                            p = p0
                            idx = i
            # 占いローラー
            else:
                if self.seer_roller == 1: #占い師ＣＯ者が3人以上なら
                    idx = 1
                    for i in self.seeList:
                        if self.base_info['statusMap'][str(i)] == 'ALIVE':
                            idx = i
                    # print('占いローラーだ！！')
                else:
                    idx = 1
                    p = -1
                    for i in range(1, 16):
                        p0 = p0_mat[i-1, 1]
                        if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                            p = p0
                            idx = i
                    #return idx
            return idx


            # else:
            #     # 一番喋ってないやつに投票する
            #     for i in range(0, 15):
            #         idx_num = self.talk_num_order_list[i]
            #         if self.base_info['statusMap'][str(idx_num)] == 'ALIVE':
            #             idx = idx_num
            #             break
            # return idx
        else:
            if self.base_info['myRole'] == "WEREWOLF":
                p0_mat = self.predicter_5.ret_pred_wx(1)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 3]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
            elif self.base_info['myRole'] == "POSSESSED":
                p0_mat = self.predicter_5.ret_pred_wx(2)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 3]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
            elif self.base_info['myRole'] == "SEER":
                p0_mat = self.predicter_5.ret_pred_wx(3)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
                        idx = i
            else:
                p0_mat = self.predicter_5.ret_pred_wx(0)
                p = -1
                idx = 1
                for i in range(1, 6):
                    p0 = p0_mat[i-1, 1]
                    if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                        p = p0
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
                # 強かったらちょっと確率上げる
                if i == (np.argsort(self.win_rate[0])) or i == (np.argsort(self.win_rate[1])) or i == (np.argsort(self.win_rate[2])):
                    p0 += 0.2
                if self.base_info['statusMap'][str(i)] == 'ALIVE' and p0 > p:
                    p = p0
                    idx = i
            return idx
        else:
            # 強いやつ噛み
            # self.strength = np.argsort(self.win_rate)
            # for i in range(0, 5):
            #     idx_num = self.strength[i] + 1
            #     if self.base_info['statusMap'][str(idx_num)] == 'ALIVE':
            #         idx = idx_num
            #         print('強いやつ噛んだ')
            #         break
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
            # 初日は勝率の高い人を占う
            if self.base_info['day'] == 0:
                self.strength = np.argsort(self.win_rate)
                idx = self.strength[0] + 1
                # print('強いやつ占った')
            else:
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

        if self.game_setting['playerNum'] == 15:
            for i in range(1, 16):
                if self.base_info['statusMap'][str(i)] == 'ALIVE':
                    self.win_rate[i-1] += 1
        else:
            for i in range(1, 6):
                if self.base_info['statusMap'][str(i)] == 'ALIVE':
                    self.win_rate[i-1] += 1
        pass



agent = SpicyPlayer(myname)

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
    # 大会時コメントアウト
    read_record()
