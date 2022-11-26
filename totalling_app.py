import pandas as pd
import csv, codecs
from collections import Counter

# coding: UTF-8
df = open(u"questionnaire01.csv", 'r', encoding='utf-8')

respondents = csv.reader(df)
header = next(csv.reader(df))

# カウンター定義
cnt = Counter()
for i in range(len(header)):
    cnt[header[i]] = Counter()

# 読み込んだファイルの各列に対しCounterを適用
for respondent in respondents:
    for i in range(len(header)):
        ress = respondent[i].split(", ")
        for res in ress:
            cnt[header[i]][res] += 1


# csv書き込み
ff = codecs.open("totalling01.csv", mode='w', encoding='utf-8')
csvWriter = csv.writer(ff)

for i in range(len(cnt)):
    # 初期化
    question = []
    answer = []
    ansNum = []

    # 質問名書き込み
    question.append(header[i])

    # keys()はdect_keys型で返るのでlistに変換
    keys = list(cnt[header[i]].keys())
    for j in range(len(cnt[header[i]])):
        # 回答書き込み
        answer.append(keys[j])
        # 回答数書き込み
        ansNum.append(cnt[header[i]][keys[j]])

    # ファイルに書き込み
    csvWriter.writerow(question)
    csvWriter.writerow(answer)
    csvWriter.writerow(ansNum)


### ここまでで、集計用のデータ('totalling01.csv')を作成した ###

### ここから、集計用データ('totalling01.csv')を再読込みし、グラフ描画したい ###

