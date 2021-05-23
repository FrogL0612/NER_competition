import tensorflow as tf
from tensorflow.keras import optimizers
from kashgari.embeddings import bert_embedding
from kashgari.tasks.labeling import bi_lstm_crf_model
from kashgari.callbacks import eval_callBack
from keras.callbacks import TensorBoard


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

embedding = bert_embedding.BertEmbedding("chinese_base")


def data_format(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.readlines()  # .encode('utf-8').decode('utf-8-sig')

    raw_X, raw_y = [], []

    for row in data:
        temp_line = row.split()
        if len(temp_line) == 2:
            raw_X.append(temp_line[0])
            raw_y.append(temp_line[1])

    sentence_data, sentence_label = [], []
    temp_sent, temp_label = [], []
    split_set = {'。', '；', '！', '？', '～', '，'}
    length = 0
    for word in range(len(raw_X)):
        if raw_X[word] in split_set:
            temp_sent.append(raw_X[word])
            temp_label.append(raw_y[word])
            sentence_data.append(temp_sent)
            sentence_label.append(temp_label)
            length = 0
            temp_sent, temp_label = [], []
        elif raw_X[word] is '…' and raw_X[word - 1] is '…':
            temp_sent.append(raw_X[word])
            temp_label.append(raw_y[word])
            sentence_data.append(temp_sent)
            sentence_label.append(temp_label)
            length = 0
            temp_sent, temp_label = [], []
        elif length == 20:
            temp_sent.append(raw_X[word])
            temp_label.append(raw_y[word])
            sentence_data.append(temp_sent)
            sentence_label.append(temp_label)
            length = 0
            temp_sent, temp_label = [], []
        else:
            length += 1
            temp_sent.append(raw_X[word])
            temp_label.append(raw_y[word])

    return sentence_data, sentence_label


def loadInputFile(file_path):
    trainingset = list()
    with open(file_path, 'r', encoding='utf8') as f:
        file_text = f.read().encode('utf-8').decode('utf-8-sig')
    datas = file_text.split('\n\n--------------------\n\n')[:-1]
    for data in datas:
        data = data.split('\n')
        content = data[1]
        trainingset.append(content)

    return trainingset


def load_test(data):
    X = []
    for article in data:
        temp = []
        for word in range(len(article)):
            temp.append(article[word])
        X.append(temp)

    article = []
    split_set = {'。', '；', '！', '？', '～', '，'}
    for articel_id in X:
        sentence_data, temp_sent = [], []
        for word in range(len(articel_id)):
            if articel_id[word] in split_set:
                temp_sent.append(articel_id[word])
                sentence_data.append(temp_sent)
                temp_sent = []
            elif articel_id[word] is '…' and articel_id[word-1] is '…':
                temp_sent.append(articel_id[word])
                sentence_data.append(temp_sent)
                temp_sent = []
            else:
                temp_sent.append(articel_id[word])
        article.append(sentence_data)

    return article


def save2csv(article, label, article_id):
    path = 'data/final_output.tsv'
    final_csv = open(path, 'a', encoding='utf-8')


    word_idx = 0
    print('aritcle_id :', article_id)
    start_to_record = False
    for sentence in range(len(label)):
        for word in range(len(label[sentence])):
            #rule based method for ID label
            check_length = len(label[sentence]) - word -1
            if check_length >= 9 and article[sentence][word] >= u'\u0041' and article[sentence][word] <= u'\u005a':
                check_id = True
                for i in range(9):
                    if article[sentence][word+1+i] >= u'\u0030' and article[sentence][word+1+i] <= u'\u0039':
                        continue
                    else:
                        check_id = False
                if check_id is True:
                    label[sentence][word] = 'B-ID'
                    for i in range(9):
                        label[sentence][word+1+i] = 'I-ID'
            #save label to tsv file
            if label[sentence][word] is not 'O' and start_to_record is False:
                record_tag = label[sentence][word][2:]
                record_word = article[sentence][word]
                start_to_record = True
                final_csv.write(str(article_id) + '\t')
                final_csv.write(str(word_idx) + '\t')
            elif start_to_record is True and label[sentence][word] is 'O':
                start_to_record = False
                final_csv.write(str(word_idx) + '\t')
                final_csv.write(record_word + '\t')
                final_csv.write(record_tag + '\n')
            elif start_to_record is True and label[sentence][word][0] is 'B':
                final_csv.write(str(word_idx) + '\t')
                final_csv.write(record_word + '\t')
                final_csv.write(record_tag + '\n')

                record_tag = label[sentence][word][2:]
                record_word = article[sentence][word]
                final_csv.write(str(article_id) + '\t')
                final_csv.write(str(word_idx) + '\t')
            elif start_to_record is True and label[sentence][word][0] is 'I':
                record_word += article[sentence][word]
            word_idx += 1

    final_csv.close()


# train_data = 'data/train_full.data'
train_data = 'data/val/train.data'
x_train, y_train = data_format(train_data)

val_data = 'data/val/val.data'
x_val, y_val = data_format(val_data)

hyper = {'layer_blstm': {'units': 140, 'return_sequences': True},
         'layer_dropout': {'rate': 0.5}, 'layer_time_distributed': {},
         'layer_activation': {'activation': 'relu'}}

model = bi_lstm_crf_model.BiLSTM_CRF_Model(embedding=embedding, hyper_parameters=hyper)

tf_board_callback = TensorBoard(log_dir='./logs', update_freq=2000)

eval_callback = eval_callBack.EvalCallBack(kash_model=model,
                                           x_data=x_val, y_data=y_val,
                                           step=1, batch_size=64)

model.build_model(x_train, y_train)
my_optimizers = optimizers.Adam(learning_rate=1e-4)
model.compile_model(optimizer=my_optimizers)

model.fit(x_train, y_train,
          x_validate=x_val, y_validate=y_val,
          epochs=50, batch_size=64,
          callbacks=[eval_callback, tf_board_callback])

test_data = loadInputFile('data/no_label.txt')
test = load_test(test_data)

path = 'data/final_output.txt'
outputfile = open(path, 'a', encoding='utf-8')
article_id = 0

report_path = 'data/final_output.tsv'
final_csv = open(report_path, 'a', encoding='utf-8')
final_csv.write('article_id' + '\t' + 'start_position' + '\t' +
                'end_position' + '\t' + 'entity_text' + '\t' +
                'entity_type' + '\n')
final_csv.close()

for article in test:
    outputfile.write('article_id:' + str(article_id))
    outputfile.write('\n')
    pred = model.predict(article)
    outputfile.write(str(pred))
    outputfile.write('\n' + '------------' + '\n')
    save2csv(article, pred, article_id)
    article_id += 1
outputfile.close()

