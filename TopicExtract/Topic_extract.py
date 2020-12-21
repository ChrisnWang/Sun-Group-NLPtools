from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from algorithm.TextCNN import TextCNN
from algorithm.RnnAtt import RnnAtt
from algorithm.RCnn import RCNN, gen_data_rcnn
import re
import jieba
import pandas as pd
import os
from common import common_function as SC
from TopicExtract.Hyper_parameter import args
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def pre_process():
    jieba.set_dictionary(args.dict_path)
    jieba.initialize()
    stopwords = [line.strip() for line in open(args.stop_words_path, encoding='UTF-8').readlines()]

    # load data
    data = pd.read_csv(args.file_path, header=0, error_bad_lines=False)
    print(data.head(10))
    train_d, valid_d = train_test_split(data, test_size=0.2, random_state=256)

    # tokenizer data
    for data in [train_d, valid_d]:
        data['review'] = data['review'].apply(lambda x: re.sub('[^\u4e00-\u9fa5]', '', x))
        data['review'] = data['review'].apply(lambda x: SC.cut_phase(x, jieba, stopwords))

    with open(args.pkl_path, 'wb') as file_write:
        pickle.dump([train_d, valid_d], file_write)

    return train_d, valid_d


def gen_data(t_list, v_list, c_list):
    tokenizer = SC.gen_tokenizer(c_list, args.max_features)
    label = LabelEncoder()

    t_x = tokenizer.texts_to_sequences(t_list)
    t_x = sequence.pad_sequences(t_x, maxlen=args.maxlen)
    t_y = to_categorical(label.fit_transform(train_data['label']), num_classes=args.num_classes)

    v_x = tokenizer.texts_to_sequences(v_list)
    v_x = sequence.pad_sequences(v_x, maxlen=args.maxlen)
    v_y = to_categorical(label.fit_transform(valid_data['label']), num_classes=args.num_classes)

    return t_x, t_y, v_x, v_y


if __name__ == '__main__':
    if not args.load_pkl:
        train_data, valid_data = pre_process()
    else:
        with open(args.pkl_path, 'rb') as f:
            train_data, valid_data = pickle.load(f)

    train_list = train_data['review'].values.flatten().tolist()
    valid_list = valid_data['review'].values.flatten().tolist()
    content_list = train_list + valid_list
    train_x, train_y, valid_x, valid_y = gen_data(train_list, valid_list, content_list)

    # model = TextCNN(args.maxlen, args.max_features, args.embedding_dims, class_num=args.num_classes)
    # model = RnnAtt(args.maxlen, args.max_features, args.embedding_dims, class_num=args.num_classes)
    model = RCNN(args.maxlen, args.max_features, args.embedding_dims, class_num=args.num_classes)
    train_x, valid_x = gen_data_rcnn(train_x, valid_x)

    model.compile('adam', 'categorical_crossentropy', metrics=['categorical_accuracy'])
    early_stopping = EarlyStopping(monitor='val_categorical_accuracy', patience=3, mode='max')
    model.fit(train_x, train_y,
              verbose=1,
              batch_size=args.batch_size,
              epochs=args.epochs,
              callbacks=[early_stopping],
              validation_data=(valid_x, valid_y))



