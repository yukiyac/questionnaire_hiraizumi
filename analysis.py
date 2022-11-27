from sklearn import preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import KMeans
from IPython.core.display import display
from sklearn import tree
import pydotplus
from IPython.display import Image
from graphviz import Digraph

# print([f.name for f in matplotlib.font_manager.fontManager.ttflist])

# フォント設定
plt.rcParams['font.family'] = 'Hiragino Sans'
# plt.rcParams['font.sans-serif'] = 'Hiragino Sans'
# plt.rcParams['axes.unicode_minus'] = False
# 太さ設定
# plt.rcParams['font.weight'] = 'bold'

# sns.set()
sns.set()

df = pd.read_csv("analysis02.csv", encoding="utf-8", header=0)

### 「住み続けたい」と考える人は、現状満足度をどう評価しているか ###
df_live = df[df["住み続けたいか"] == 5]  # 便宜上、「住み続けたい」としているデータのみ抽出
df_live = df_live.drop("住み続けたいか", axis=1)  # 「住み続けたいか」の列を削除
# df_live = df_live.reset_index(drop=True) # インデックス番号を振り直し

df_live.describe()  # データの標準化

ss = preprocessing.StandardScaler()
df_live_s = pd.DataFrame(ss.fit_transform(df_live))

df_live_s.columns = df_live.columns  # 列名が消えるので、振り直し

print(df_live_s)

### ここまでで前処理が完了 ###

### クラスター分析に挑戦 ###

df_live_s_hclust = linkage(df_live_s, metric="euclidean", method="ward")
plt.figure(figsize=(12, 8))
dendrogram(df_live_s_hclust)
plt.savefig('figure_1.png')
plt.show()

km = KMeans(n_clusters=4, random_state=42)  # クラスター数を４に設定

df_live_s_ar = df_live_s.values
# display(df_live_s_ar)

# KMeansを適用した結果のグルーピングの配列が出力として渡される
df_live_s_ar_pred = km.fit_predict(df_live_s_ar)

# 割り振られた結果の元のデータにクラスターIDとして一列追加する
df_live_clust = df_live[:]
df_live_clust["cluster_ID"] = df_live_s_ar_pred

print(df_live_clust.dtypes)

df_live_clust["cluster_ID"] = df_live_clust["cluster_ID"].astype("category")
# print(df_live_clust.dtypes)

# print(df_live_clust["cluster_ID"].value_counts())

### ここからいよいよ各クラスターの分析 ###

df_live_clust = df_live_clust[:].astype("category")
# print(df_live_clust.dtypes)

dummy_list = list(df_live_clust.columns)[0:-1]
# print(dummy_list)

df_live_clust_dmy = pd.get_dummies(df_live_clust, columns=dummy_list)
df_live_clust_dmy.head(3)
df_live_clust_dmy_gp = df_live_clust_dmy.groupby("cluster_ID")

# グループ別に書く設問の回答者数の合計を出す
df_live_clust_dmy_gp_g = df_live_clust_dmy_gp.sum().T
display(df_live_clust_dmy_gp_g)

# グラフ化
df_live_clust_dmy_gp_g.style.bar(color="#4285F4")

plt.figure(figsize=(12, 8))
sns.clustermap(df_live_clust_dmy_gp_g, cmap="viridis")
plt.savefig('figure_2.png')
plt.show()

# スプレッドシートに吐き出して条件付書式で見られるようにファイルを出力
df_live_clust_dmy_gp_g.to_csv("df_live_clust_dmy_gp_g.csv")

### ここまででクラスター分析終了 ###

### ここから決定木分析に挑戦 ###

# 先程のデータをラベルとデータに分けて、どちらもnumpyの配列にするぜ
y = np.array(df_live_clust["cluster_ID"].values)
X = df_live_clust.drop("cluster_ID", axis=1).values

# 仮の数字で段階を決める（ここでは4にした）
dtree = tree.DecisionTreeClassifier(max_depth=4)
dtree = dtree.fit(X, y)

# モデル精度の確認
dtree_pred = dtree.predict(X)
# display(dtree_pred)

sum(dtree_pred == y) / len(y)

# 列名を代入、ラベルデータはastypeを使って文字列（string）に変更
dot_data = tree.export_graphviz(
    dtree, out_file=None, feature_names=df_live_clust.columns[0:-1], class_names=y.astype("str"))

# 決定機を日本語で出力する　←画像は出力されるが実行時に「missing from current font.」のエラーが出る
graph = pydotplus.graph_from_dot_data(dot_data)
graph.set_fontname('Hiragino Sans')
for node in graph.get_nodes():
    node.set_fontname('Hiragino Sans')
for e in graph.get_edges():
    e.set_fontname('Hiragino Sans')

graph.write_png("dtree.png")
Image(graph.create_png())

df_live_clust_dmy[df_live_clust_dmy["cluster_ID"] == 3].sum().head(10)
