import MeCab

text = "天気がいいから、散歩しましょう"
mecab_tagger = MeCab.Tagger("-Owakati")
print(mecab_tagger.parse(text))
print(mecab_tagger.parse(text).split()[:-1])