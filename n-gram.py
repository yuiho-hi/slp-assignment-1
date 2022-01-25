import os.path
import math
import numpy as np



number = 3 #n-gramのnを決定
discounting = 0.75 #Kneser-Ney smoothingのdiscount

counts = {} #各単語の出現回数を保存するための辞書
context_counts = {} #先行する単語の出現回数を保存するための辞書
following_words_types = {} #後行する単語の種類数を保存するための辞書　Kneser-Ney smoothingのλ計算用
preceding_words_types = {} #先行する単語の種類数を保存するための辞書　Kneser-Ney smoothingのCkn(continuationcount)計算用
dirname = os.getcwd() #現在のファイルがあるフォルダのパスを取得

#-------------------------------------------------------
#n-gram取得
#-------------------------------------------------------
   
def n_gram(str, n): #n-gram
    #引数: (word: 1文、n: gram数)
    return [str[i:i+n] for i in range(len(str) - n + 1)]

#-------------------------------------------------------
#train用プログラム
#-------------------------------------------------------
        
def all_context(words, n): #設定n_gram以下の単語組み合わせを全て取得する関数
    #引数: (word: 1文、n: gram数)
    for gram in range(1, n+1):
        
        if len(words) >= gram:
            str_words = n_gram(' '.join(words).split(), gram)
        else:
            #print(f'words: {words}')
            str_words = n_gram(' '.join(words).split(), len(words))
        #print(f'str_words: {str_words}')
        '''
        例: unigram
        str_words: [['<s>'], ['The'], ['machine-learning'], ['paradigm'], ['calls'], ['instead'], ['for'], ['using'], 
        ['general'], ['learning'], ['algorithms'], ['—'], ['often'], ['although'], ['not'], ['always'], ['grounded'], 
        ['in'], ['statistical'], ['inference'], ['—'], ['to'], ['automatically'], ['learn'], ['such'], ['rules'], 
        ['through'], ['the'], ['analysis'], ['of'], ['large'], ['corpora'], ['of'], ['typical'], ['real-world'], 
        ['examples'], ['</s>']]

        bigram
        str_words: [['<s>', 'The'], ['The', 'machine-learning'], ['machine-learning', 'paradigm'], ['paradigm', 'calls'], 
        ['calls', 'instead'], ['instead', 'for'], ['for', 'using'], ['using', 'general'], ['general', 'learning'], 
        ['learning', 'algorithms'], ['algorithms', '—'], ['—', 'often'], ['often', 'although'], ['although', 'not'], 
        ['not', 'always'], ['always', 'grounded'], ['grounded', 'in'], ['in', 'statistical'], ['statistical', 'inference'], 
        ['inference', '—'], ['—', 'to'], ['to', 'automatically'], ['automatically', 'learn'], ['learn', 'such'], ['such', 'rules'], 
        ['rules', 'through'], ['through', 'the'], ['the', 'analysis'], ['analysis', 'of'], ['of', 'large'], ['large', 'corpora'], 
        ['corpora', 'of'], ['of', 'typical'], ['typical', 'real-world'], ['real-world', 'examples'], ['examples', '</s>']]
        '''
        for i in range(len(str_words)):
            #print(f'join(str_words[i]): {" ".join(str_words[i])}') 
            
            if (' '.join(str_words[i])) in counts: #各n-gramにおいて単語群の出現回数を格納する
                counts[' '.join(str_words[i])] += 1 #C(Wi-n+1 W) に相当
            else: #単語が初めて出現した時
                counts[' '.join(str_words[i])] = 1 #初回カウント定義

                if (' '.join(str_words[i][:gram-1])) in following_words_types: #後行する単語群の種類数を格納する λの{V:C(Wi-n+1 Wi-1 V)>0} 計算用
                    following_words_types[' '.join(str_words[i][:gram-1])] += 1 #{V:C(Wi-n+1 Wi-1 V)>0} に相当
                else:
                    following_words_types[' '.join(str_words[i][:gram-1])] = 1 #初回カウント定義

                if gram >= 2: #bigram以上の時
                    if (' '.join(str_words[i][1:]) + '\t' + str(gram - 1)) in preceding_words_types: #先行する単語群の種類数を格納する {V:C(V Wi-n+1 Wi)>0} 計算用
                        preceding_words_types[' '.join(str_words[i][1:]) + '\t' + str(gram - 1)] += 1 #{V:C(V Wi-n+1 Wi)>0} に相当
                    else:
                        preceding_words_types[' '.join(str_words[i][1:]) + '\t' + str(gram - 1)] = 1 #初回カウント定義

                    if gram == 2: #bigramの時
                        if (str_words[i][0] + '\t' + '0') not in preceding_words_types: #先行する単語群の種類数を格納する ΣW'{V:C(V W')>0} 計算用
                            preceding_words_types[str_words[i][0] + '\t' + '0'] = 1 #重複判定用
                            preceding_words_types[''] += 1 #ΣW'{V:C(V W')>0} に相当

                    if gram >= 3: #trigram以上の時
                        if (' '.join(str_words[i][:gram - 1])) not in preceding_words_types: #先行する単語群の種類数を格納する ΣW'{V:C(V Wi-n+1 W')>0} 計算用
                            preceding_words_types[' '.join(str_words[i][:gram - 1])] = 1
                            if (' '.join(str_words[i][1:gram - 1]) + '\t' + str(gram - 1)) in preceding_words_types:
                                preceding_words_types[' '.join(str_words[i][1:gram - 1]) + '\t' + str(gram - 1)] += 1 #ΣW'{V:C(V Wi-n+1 W')>0} に相当
                            else:
                                preceding_words_types[' '.join(str_words[i][1:gram - 1]) + '\t' + str(gram - 1)] = 1 #初回カウント定義

            if ' '.join(str_words[i][:gram-1]) in context_counts: #先行する単語群の出現回数を格納する
                context_counts[' '.join(str_words[i][:gram - 1])] += 1 #C(Wi-n+1 Wi-1) に相当
            else:
                context_counts[' '.join(str_words[i][:gram - 1])] = 1 #初回カウント定義
            
            if gram == 1: #unigramの時、確率算出のため全単語数を計測する
                context_counts[""] += 1

def load_train(tgtfile, n): #ファイル読み込み用関数
    #引数: (tgtfile: train指定ファイル、n: gram数)
    with open(tgtfile, mode='rt') as tgt_f:
        save_list = []
        context_counts[""] = 0
        preceding_words_types[''] = 0
        for line in tgt_f:
            words = line.translate(str.maketrans({',': '', '.': ''})).split()
            words.insert(0, "<s>")
            words.append("</s>") #最初と最後に<s>・</s>を追加
            #print(f'words: {words}')
            '''
            例:
            words: ['<s>', 'The', 'machine-learning', 'paradigm', 'calls', 'instead', 'for', 'using', 'general', 'learning', 
            'algorithms', '—', 'often', 'although', 'not', 'always', 'grounded', 'in', 'statistical', 'inference', '—', 'to', 
            'automatically', 'learn', 'such', 'rules', 'through', 'the', 'analysis', 'of', 'large', 'corpora', 'of', 'typical', 
            'real-world', 'examples', '</s>']
            '''
            all_context(' '.join(words).split(), n)

        for ngram, count in counts.items():
            words = ngram.split()
            words.pop()
            context = ' '.join(words)
            
            probability = counts[ngram] / context_counts[context]
            save_list.append(str(ngram) + '\t' + str(probability))

        #print(f'following_words_types: {following_words_types}')
        savefile_en(save_list, n)



        for_confirm = [] #デバッグ用

        '''for ngram, count in following_words_types.items():
            for_confirm.append('following: ' + str(ngram) + '\t' + str(count))

        savefile_en(for_confirm, 'following')#'''
        
        '''for_confirm = [] #デバッグ用

        for ngram, count in preceding_words_types.items():
            for_confirm.append('preceding: ' + str(ngram) + '\t' + str(count))

        savefile_en(for_confirm, 'mini')#'''

#-------------------------------------------------------
#各n-gramと確率をファイルに保存
#-------------------------------------------------------

def savefile_en(list, n): #パラメータ保存用関数
    #引数: (list: 保存対象リスト、n: gram数)
    np.savetxt(os.path.join(dirname, f"model-train-{n}gram.txt"), list, fmt="%s")

#-------------------------------------------------------
#test用プログラム
#-------------------------------------------------------

test_vocaburary = {} #testデータの語彙数把握用

def normalizing_constant(str_words, gram, d): #Kneser-Ney smoothingのλ計算関数
    #引数: (str_words: 対象となるn-gramに分けた1文　範囲[:gram - 1]、gram: gram数, d: discounting)¥
    return (d / context_counts[' '.join(str_words)]) * following_words_types[' '.join(str_words)]

def probability_KN(words, gram, pro_KN, total, d):
    #引数: (words: 対象となる1文、gram: gram数, pro_KN: probabilty_KNの値、total: testデータの語彙数、d: discounting)
    if len(words) >= gram:
        str_words = n_gram(' '.join(words).split(), gram)
    else:
        str_words = n_gram(' '.join(words).split(), len(words))
    #print(f'str_words: {str_words}')
    #print(f'gram: {gram}')
    for i in range(len(str_words)):
        if gram > 1:
            if (' '.join(str_words[i]) in counts) and (' '.join(str_words[i][:gram - 1]) in context_counts):
                if gram == number: #for the highest order         
                    pro_KN += max(counts[' '.join(str_words[i])] - d, 0) / context_counts[' '.join(str_words[i][:gram - 1])] + normalizing_constant(str_words[i][:gram - 1], gram, d) * probability_KN(words, gram - 1, pro_KN, total, d)
                    #1項目: max(C(Wi-n+1:i) - d, 0) / C(Wi-n+1:i-1)　に相当
                    #2項目: λ(Wi-n+1:i-1) * Pkn(Wi|Wi-n+2:i-1)　に相当
                else: #for lower orders 
                    pro_KN += max(preceding_words_types[' '.join(str_words[i][1:]) + '\t' + str(gram - 1)] - d, 0) / preceding_words_types[' '.join(str_words[i][1:gram]) + '\t' + str(gram)] + normalizing_constant(str_words[i][:gram - 1], gram, d) * probability_KN(words, gram - 1, pro_KN, total, d)
                    #1項目: max(|{V:C(V Wi-n+2 : Wi)}| - d, 0) / ΣW'|{V:C(V Wi-n+2 : Wi-1 W')}|　に相当
                    #2項目: λ(Wi-n+1:i-1) * Pkn(Wi|Wi-n+2:i-1)　に相当
            else: #未知語だった場合
                #pro_KN += d/total + normalizing_constant(str_words[i][:gram - 1], gram, d) * probability_KN(words, gram - 1, pro_KN, total, d)
                pro_KN += d/total + probability_KN(words, gram - 1, pro_KN, total, d)     
            
            return pro_KN

        else: #unigram用の処理
            if ' '.join(str_words[i]) in counts:
                if number == 1: #元からunigramだった場合
                    pro_KN += max(counts[' '.join(str_words[i])] - d, 0) / context_counts[''] + d/total
                    #1項目: max(C(Wi) - d, 0) / ΣW' C(W')　に相当
                    #2項目: d/V　に相当
                elif ' '.join(str_words[i]) != '<s>':
                    pro_KN += max(preceding_words_types[' '.join(str_words[i])  + '\t' + str(0)] - d, 0) / preceding_words_types[''] + d/total
                    #1項目: max(|{V:C(V Wi)}| - d, 0) / ΣW'|{V:C(V W')}|　に相当
                    #2項目: d/V　に相当
            else: #未知語だった場合
                pro_KN += d/total

            return pro_KN

def load_test(test_file, gram, d):
    #引数: (test_file: test指定ファイル、gram: gram数、d: discounting)
    with open(test_file, mode='rt') as f: #testデータ全単語計測
        total_count = 0 #testデータの語彙数
        W = 0 #entropy算出用 testデータ全単語数
        H = 0 #entropy算出用 各確率のlogを加算
        for line in f:
            words = line.translate(str.maketrans({',': '', '.': ''})).split()
            words.insert(0, "<s>")
            words.append("</s>")

            for i in range(len(words)):
                W += 1
                if words[i] not in test_vocaburary:
                    test_vocaburary[words[i]] = 1
                    total_count += 1

        #print(f'test_vocaburary: {test_vocaburary}')
        #print(f'total_count: {total_count}')
        
    pro_KN_save = []
    with open(test_file, mode='rt') as f:
        for line in f:
            pro_KN = 0
            words = line.translate(str.maketrans({',': '', '.': ''})).split()
            words.insert(0, "<s>")
            words.append("</s>")

            pro_KN = probability_KN(' '.join(words).split(), gram, pro_KN ,total_count, d)
            #print(f'pro_KN: {pro_KN}')
            H += - math.log2(pro_KN)
            #print(f'H: {H}')
            pro_KN_save.append(H)
            
        #print(f'W: {W}')
        print(f'entropy = {H / W}')

    savefile_en(pro_KN_save, 'probability')#'''


load_train(os.path.join(dirname, "wiki-en-train.txt"), number) #学習　trainデータとn-gram数を渡す
#load_train(os.path.join(dirname, "wiki-en-train-mini.txt"), number) #デバッグ用
load_test(os.path.join(dirname, "wiki-en-test.txt"), number, discounting) #テスト　testデータとn-gram数とdiscountingを渡す
#load_test(os.path.join(dirname, "wiki-en-test-mini.txt"), number, discounting) #デバッグ用




