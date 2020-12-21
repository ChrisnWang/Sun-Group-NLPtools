from keras.preprocessing.text import Tokenizer


def gen_tokenizer(dict_list, num_words):
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(dict_list)
    return tokenizer


def move_stopwords(sentence_list, stopwords_list):
    out_list = []
    for word in sentence_list:
        if word not in stopwords_list:
            if word != '\t':
                out_list.append(word)
    return out_list


def cut_phase(phase, jieba, stopwords):
    seg_list = jieba.lcut(phase)
    seg_list = move_stopwords(seg_list, stopwords)
    token_list = ' '.join(seg_list)
    return token_list
