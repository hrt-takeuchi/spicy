# Spicy

### DNNで作ったモデル人狼推定を行ってます


### 実装
	占い師の時：
		初日は勝率が高いAgentを占う
		黒に投票（投票合わせ）
	霊媒師の時：
		霊媒師は黒確定のちCO
	人狼の時：
		PP
	狂人の時：
		PP
		ランダムでCO（たまにいる狂った人間らしさ）
	その他の時：
		PP対策
		占い師が３人以上出たら占い師ローラー
	村人側の時に自分への黒出し占い師へ投票（投票合わせ）
	投票逃れ



#### パワープレイ
	生存リストが３の時
	人狼→CO
	狂人→CO
	村側→人狼CO
	VOTE→村人


