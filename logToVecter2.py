import  multiprocessing

def createdata(saveFile):

    for j in range(779):
            #778:100
        for k in range(100):
            f = open('/home/spicy/Desktop/jupyter/jupyter_memo/log/'+str(j).zfill(3)+'/'+str(k).zfill(3)+'.log') #ファイル読み込み
            lines =f.readlines()# 1行毎にファイル終端まで全て読む(改行文字も含まれる)
            f.close()
            # lines: リスト。要素は1行の文字列データ

            # COしてる占い師
            numCoSeer = 0
            # Seerなら１そうでなければ０
            SeerOrNot = [0 for i in range(15)]
            # 処刑なら1, 襲撃なら2
            raidedOrNot = [0 for i in range(15)]
            # talk数
            numOfTalk = [0 for i in range(15)]

            # 日にち
            date = 1

            # 人間推定の数
            estimate_human = [0 for i in range(15)]
            # 同意した数
            agree_num = [0 for i in range(15)]

            # 人狼推定の数
            estimate_wolf = [0 for i in range(15)]
            # 否定した数
            disagree_num = [0 for i in range(15)]

            # 規定
            info = [[0 for i in range(11)] for t in range(15)]

#             for i in range(15) :
#                 info[i][0] = 0 # 現在何日か
#                 info[i][1] = 0 # 現在COしている占い師プレイヤーの数
#                 info[i][2] = 0 # 人間判定された数（累計）
#                 info[i][3] = 0 # 人狼判定された数（累計）
#                 info[i][4] = 0 # 報告した人間判定の数
#                 info[i][5] = 0 # 報告した人狼判定の数
#                 info[i][6] = 0 # 生きていたら0,死んでいたら１
#                 info[i][7] = 0 # 処刑なら1, 襲撃なら2
#                 info[i][8] = 0 # 誰かに村人陣営であると推定した数 + AGREE発言の数（累計）
#                 info[i][9] = 0 # 誰かに人狼陣営であると推定した数 + DISAGREE発言の数（累計）
#                 info[i][10] = 0 # 会話数

            for line in lines:
                newLine = line.replace('\n','').replace(' ', ',').split(',')
                if newLine[0] == 0:
                    continue
				# 日にちが変わったら前日までのinfoをinfoListに格納
                if int(newLine[0]) > 1 and int(newLine[0]) != date :
                    for m in range(15):
                        info[m][1] = numCoSeer
                        info[m][0] = date
                    # 前日までの情報を保持
                    tempInfo = [[0]*11] *15
                    for a in range(15):
                        for b in range(11):
                            tempInfo[a][b] = info[a][b]

                    file = open(saveFile, 'a')    #追加書き込みモードでオープン
                    file.writelines(str(info)[1:-1]+",")
                    file.close()
                    if int(newLine[0]) != 1:
                        date = int(newLine[0])


                # status
                if newLine[1] == "status":

                    i = int(newLine[2]) - 1 # AgentID
                    # 生きてたら０
                    if newLine[4] == "ALIVE":
                        info[i][6] = 0
                    elif newLine[4] == "DEAD":
                        info[i][6] = 1
                # TALK
                if newLine[1] == "talk":

                    i = int(newLine[4]) - 1 # AgentID
                    if newLine[5] != "Skip" and newLine[5] != "Over" :
                        numOfTalk[i] += 1  # talk数を加算
                    info[i][10] = numOfTalk[i]
                    if newLine[5]=="COMINGOUT"  and newLine[7] =="SEER":
                        SeerOrNot[i] = 1
                        numCoSeer += 1
                        info[i][1] = numCoSeer

                    # 受けた占い人狼判定数
                    # 占い師で人狼判定を出した数
                    if SeerOrNot[i] == 1 and newLine[5] == "DIVINED" and newLine[7] == "WEREWOLF" :
                        target = newLine[6].replace('Agent[','').replace(']','')
                        info[int(target)-1][3] += 1
                        info[i][5] += 1


                    # 受けた占い人間判定数
                    # 占い師で人間判定を出した数
                    if SeerOrNot[i] == 1 and newLine[5] == "DIVINED" and newLine[7] != "WEREWOLF":
                        target = newLine[6].replace('Agent[','').replace(']','')
                        info[int(target)-1][2] += 1
                        info[i][4] += 1


                    # 肯定的意見の数
                    if newLine[5]=="ESTIMATE"  and newLine[7] !="WAREWOLF":
                        estimate_human[i] += 1
                    if newLine[5]=="AGREE":
                        agree_num[i] += 1

                    info[i][8] = estimate_human[i] + agree_num[i]



                    # 否定的意見の数
                    if newLine[5]=="ESTIMATE"  and newLine[7] =="WAREWOLF":
                        estimate_wolf[i] += 1
                    if newLine[5]=="DISAGREE":
                        disagree_num[i] += 1
                    info[i][9] = estimate_wolf[i] + disagree_num[i]

                # 襲撃
                if newLine[1] == "attack" and newLine[3] == "true":
                    i = int(newLine[2]) - 1
                    raidedOrNot[i] = 2
                elif newLine[1] == "execute":
                    i = int(newLine[2]) - 1
                    raidedOrNot[i] = 1
        # print(str(j) + "フォルダ/" + str(k) + "週目")

    # マルチプロセス処理
    if __name__ == '__main__':
        jobs = []
        p = multiprocessing.Process(target=createdata)
        jobs.append(p)
        p.start()
