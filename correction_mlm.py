# @Author:sunshine
# @Time  : 2020/5/12 上午9:17

import json
from bert4keras.tokenizers import load_vocab, Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding
from bert4keras.optimizers import AdaFactor
from keras.layers import Lambda
from keras.models import Model
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from tqdm import tqdm

max_len = 64
config_path = './chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = './chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = './chinese_L-12_H-768_A-12/vocab.txt'

train_data = json.load(open('./data/train_data.json', 'r', encoding='utf-8'))
valid_data = json.load(open('./data/valid_data.json', 'r', encoding='utf-8'))

# 加载精简词表
token_dict, keep_words = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict, do_lower_case=True)


class MyDataGenerator(DataGenerator):
    def __iter__(self, random=True):
        """
        单条样本格式: [cls]错误词汇[sep][mask][mask]..[sep]
        :param random:
        :return:
        """
        batch_tokens_ids, batch_segment_ids, batch_right_token_ids = [], [], []
        for is_end, D in self.sample(random):
            wrong, right = D
            right_token_ids, _ = tokenizer.encode(first_text=right)
            wrong_token_ids, _ = tokenizer.encode(first_text=wrong)

            token_ids = wrong_token_ids
            token_ids += [tokenizer._token_mask_id] * max_len
            token_ids += [tokenizer._token_end_id]

            segemnt_ids = [0] * len(token_ids)

            batch_tokens_ids.append(token_ids)
            batch_segment_ids.append(segemnt_ids)
            batch_right_token_ids.append(right_token_ids[1:])

            if len(batch_tokens_ids) == self.batch_size or is_end:
                batch_tokens_ids = sequence_padding(batch_tokens_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_right_token_ids = sequence_padding(batch_right_token_ids, max_len)

                yield [batch_tokens_ids, batch_segment_ids], batch_right_token_ids
                batch_tokens_ids, batch_segment_ids, batch_right_token_ids = [], [], []


# 构建模型
bert_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    keep_tokens=keep_words
)

output = Lambda(lambda x: x[:, 1:max_len + 1])(bert_model.output)
model = Model(bert_model.input, output)


def masked_cross_entropy(y_true, y_pred):
    """交叉熵作为loss，并mask掉padding部分的预测
    """
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy


model.compile(loss=masked_cross_entropy, optimizer=AdaFactor(learning_rate=1e-3))
model.summary()


def ge_answer(wrong):
    """
    解码
    :param wrong:
    :return:·
    """
    wrong_token_ids, _ = tokenizer.encode(wrong)
    token_ids = wrong_token_ids + [tokenizer._token_mask_id] * max_len + [tokenizer._token_end_id]
    segemnt_ids = [0] * len(token_ids)
    probas = model.predict([np.array([token_ids]), np.array([segemnt_ids])])[0]
    proba_ids = probas.argmax(axis=1)
    useful_index = proba_ids[np.where(proba_ids != 3)]
    if any(useful_index):
        answer = tokenizer.decode(useful_index)
    else:
        answer = tokenizer.decode(proba_ids[:len(wrong)])
    return answer


def evalute(valid_data):
    X, Y = 1e-10, 1e-10
    for item in tqdm(valid_data):
        wrong, right = item
        pred = ge_answer(wrong)
        X += pred == right
        Y += 1
    precision = X / Y
    return precision


class Evaluator(Callback):
    def __init__(self):
        self.lowest = 1e10
        self.f1 = 1e-10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            # model.save('best_mlm_model_epoch=14_plus.h5')


if __name__ == '__main__':
    
    # 训练模型
    evaluator = Evaluator()
    
    # train_generator = MyDataGenerator(train_data, batch_size=8)
    # model.load_weights('/home/tongtong/python_project/nlp_project/chinese_spelling_correction/best_mlm_model_epoch=14_plus.h5')
    # model.fit_generator(
    #     train_generator.forfit(),
    #     steps_per_epoch=len(train_generator),
    #     epochs=2,
    #     callbacks=[evaluator]
    # )
  
    # predict
    model.load_weights('best_mlm_model_epoch=14_plus.h5')
    wrong = '自然预言处理真太是有趣啦'
    result = ge_answer(wrong)
    print(result)

    # test

    # detection
    # num_correct=100
    # num=223
    # model.load_weights('/home/tongtong/python_project/nlp_project/chinese_spelling_correction/best_mlm_model_epoch=29.h5')
    # test_data = json.load(open('data/test_data.json', 'r', encoding='utf-8'))
    # #[['wrong','right'],[]...]  length=223
    # print(test_data)
    # test_x=[]
    # test_y_bool=[0]*num_correct+[1]*(num-num_correct)  # 100 correct, 123 wrong true label
    # test_y=[]
    # pred_y_bool=[]
    # correct=0
    # for i in range(num_correct):
    #     test_x.append(test_data[i][1])
    #     test_y.append(test_data[i][1])
    # for i in range(num_correct,num):
    #     test_x.append(test_data[i][0])
    #     test_y.append(test_data[i][1])
    # for i,sen in enumerate(test_x):
    #     print(sen)
    #     result=ge_answer(sen)
    #     if(result==test_x[i]):  # detect nothing
    #         pred_y_bool.append(0)
    #     else:
    #         pred_y_bool.append(1) #detect something
    #         if result==test_y[i]:
    #             correct+=1
    # #detection
    # TP=FP=TN=FN=0
    # for i in range(num):
    #     print(i)
    #     if pred_y_bool[i]==1:
    #         if test_y_bool[i]==1:
    #             TP+=1
    #         else:
    #             FP+=1
    #     else:
    #         if test_y_bool[i]==0:
    #             TN+=1
    #         else:
    #             FN+=1
    # acc=TP/(TP+FP)
    # recall=TP/(TP+FN)
    # f1=(2*acc*recall)/(acc+recall)
    # print("acc",acc,"recall",recall,"f1",f1)
    # # correction 
    # correct_rate=correct/(num-num_correct)
    # print("correct_rate",correct_rate)


            
        

    # print(1)