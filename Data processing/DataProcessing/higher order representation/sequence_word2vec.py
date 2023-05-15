import re
from gensim.models import Word2Vec


import csv
csv.field_size_limit(500 * 1024 * 1024)


def read_csv(save_list, file_name):
    csv_reader = csv.reader(open(file_name))
    for row in csv_reader:
        save_list.append(row)
    return


def store_csv(data, file_name):
    with open(file_name, "w", newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)
    return


# https://blog.csdn.net/sinat_26917383/article/details/69803018

MyMiRBase = []
read_csv(MyMiRBase, 'phage host sequence.csv')  ###################################################

miRNACorpus = []
counter = 0

# while counter < len(MyMiRBase):
#     row = re.findall(r'.{1}', MyMiRBase[counter][1])    # 1-mer https://www.jb51.net/article/139207.htm 学一学正则表达式
#     miRNACorpus.append(row)
#     counter = counter + 1


while counter < len(MyMiRBase):
    row = re.findall(r'.{1}', MyMiRBase[counter][1])    # 1-mer https://www.jb51.net/article/139207.htm 学一学正则表达式
    miRNACorpus.append(row)
    counter = counter + 1

# model = Word2Vec(miRNACorpus, min_count=1, size=64)
model = Word2Vec(miRNACorpus, min_count=5, sg=1, size=32)


miRNAEmbedding = []
counter = 0
while counter < len(list(model.wv.vocab)):
    row = []
    row.append(list(model.wv.vocab)[counter])
    row.extend(model[list(model.wv.vocab)[counter]])
    miRNAEmbedding.append(row)
    counter = counter + 1

store_csv(miRNAEmbedding, 'phagehost988_Embedding32.csv')

# model = gensim.models.Word2Vec.load('miRNAmodel')  # 读取模型
model.save('drug_model')   # 保存模型