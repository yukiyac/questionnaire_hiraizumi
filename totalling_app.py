import pandas as pd
import csv, codecs
from collections import Counter

### 不要データを削除して再保存 ###

# coding: UTF-8
df = pd.read_csv("questionnaire01.csv")
# 先頭列（タイムスタンプ列を削除して再保存)
df.iloc[:, 1:].to_csv("questionnaire02.csv")


### 再保存したcsvを読み込み、集計用データを作成 ###

df2 = open(u"questionnaire02.csv", 'r', encoding='utf-8')

respondents = csv.reader(df2)
header = next(csv.reader(df2))

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
ff = codecs.open("totalling01.csv", mode='w', encoding='shift_jis')
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

