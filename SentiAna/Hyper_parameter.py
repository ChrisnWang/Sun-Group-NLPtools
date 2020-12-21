import argparse

parser = argparse.ArgumentParser()

# file path
parser.add_argument('--load_pkl',  default=True, type=bool, required=False,
                    help='')

parser.add_argument('--pkl_path',  default=r'F:\python_code\NLPTools\pkl\sentence.pkl',
                    type=str, required=False,
                    help='')

parser.add_argument('--file_path',
                    default=r'F:\python_code\NLPTools\data\weibo_senti_100k.csv',
                    type=str, required=False,
                    help='')

parser.add_argument('--dict_path',  default=r'F:\python_code\NLPTools\dicts\dict_big.txt',
                    type=str, required=False,
                    help='')

parser.add_argument('--stop_words_path',  default=r'F:\python_code\NLPTools\dicts\stop_words.txt',
                    type=str, required=False,
                    help='')

# train mode
parser.add_argument('--batch_size',  default=10, type=int, required=False,
                    help='')

parser.add_argument('--epochs',  default=2, type=int, required=False,
                    help='')

parser.add_argument('--num_classes',  default=2, type=int, required=False,
                    help='')

# common parameters
parser.add_argument('--max_features', default=10000, type=int, required=False,
                    help='')

parser.add_argument('--maxlen',  default=40, type=int, required=False,
                    help='')

parser.add_argument('--embedding_dims',  default=50, type=int, required=False,
                    help='')

parser.add_argument('--last_activation',  default='sigmoid', type=str, required=False,
                    help='')

# TextCNN
parser.add_argument('--kernel_sizes',  default=[3, 4, 5], type=list, required=False,
                    help='')

# RnnAtt
parser.add_argument('--rnn_out_dims',  default=100, type=int, required=False,
                    help='')

parser.add_argument('--att_out_dims',  default=128, type=int, required=False,
                    help='')

parser.add_argument('--dropout_ratio',  default=0.5, type=int, required=False,
                    help='')

# RCnn
parser.add_argument('--rnn_out_dims',  default=100, type=int, required=False,
                    help='')

parser.add_argument('--conv_size',  default=50, type=int, required=False,
                    help='')

parser.add_argument('--dropout_ratio',  default=0.5, type=int, required=False,
                    help='')

args = parser.parse_args()
