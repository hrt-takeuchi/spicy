import  multiprocessing
import glob
from collections import OrderedDict
import os
from datetime import datetime
def real_vote_rate(agent,savefile):
    
    vote_num = 0
    hit_num = 0

    # logファイルだいたい100個くらい
    for k in range(500):
        print(str(k)+ 'ファイル目')

        try:
            f = open('../log/'+str(k).zfill(3)+'.log') #ファイル読み込み
        except:
            continue
        lines =f.readlines()# 1行毎にファイル終端まで全て読む(改行文字も含まれる)
        f.close()
        # lines: リスト。要素は1行の文字列データ
        # 投票相手
        vote = 0

        selfid = 0
        # 人狼リスト
        wolf_list = []
        # トリガー
        torriger = 0
        for line in lines:
            newLine = line.replace('\n','').replace(' ', ',').split(',')
            
            if int(newLine[0]) == 0 and newLine[1] == 'status':
                # 人狼リスト作成
                if newLine[3] == 'WEREWOLF':
                    wolf_list.append(newLine[2])
                ## 自分のid
                if newLine[5] == agent:
                    selfid = newLine[2]
                if newLine[5] == agent and (newLine[3] == "WEREWOLF" or newLine[3] == "POSSESSED"):
                    break


            
            if newLine[1] == 'vote' and newLine[2] == selfid: 
                vote_num += 1
                vote = newLine[3]
                print(vote)
                if vote in wolf_list:
                    hit_num += 1
        print(wolf_list)
    hitracio = (hit_num / vote_num )* 100
    text = '総投票数は' + str(vote_num) + '回' + '\n' + '正解数は' + str(hit_num) + '回' + '\n' + '投票正解率は' + str('{:.1f}'.format(hitracio)) + '%' + '\n' 
    file = open(savefile, 'a')    #追加書き込みモードでオープン
    file.writelines(text)
    file.close()


# 参照エージェント名
agent = 'voteSample'

# マルチプロセス処理
if __name__ == '__main__':
    now = datetime.now()
    savefile = '{0:%Y%m%d-%H:%M:%s}w_rate.txt'.format(now)
    jobs = []
    p = multiprocessing.Process(target=real_vote_rate(agent,savefile))
    jobs.append(p)
    p.start()
