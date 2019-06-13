
# coding: utf-8

# In[242]:


import os
import json
import nltk
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
from pylab import rcParams

import torch
import torch.nn.functional as F
from pytorch_pretrained_bert import tokenization, BertTokenizer, BertModel, BertForMaskedLM, BertForPreTraining, BertConfig
from examples.extract_features import *


# In[274]:
class Args:
    def __init__(self):
        pass
    
args = Args()
args.no_cuda = False #不用GPU

CONFIG_NAME = 'bert_config.json'
BERT_DIR = '/nas/pretrain-bert/pretrain-pytorch/bert-base-uncased'
config_file = os.path.join(BERT_DIR, CONFIG_NAME)
config = BertConfig.from_json_file(config_file)

try:
    tokenizer = BertTokenizer.from_pretrained(os.path.join(BERT_DIR, 'vocab.txt'))#do_lower_case：在标记化时将文本转换为小写。默认= True
except:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer.tokenize = nltk.word_tokenize

model = BertForMaskedLM.from_pretrained(BERT_DIR)
device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
_ = model.to(device)
_ = model.eval()

input_ids_sen,input_type_ids_sen,in_sentence,sentences,entire_ids,entire_type_ids = [],[],[],[],[],[]
suggestions = {} #外部变量，需要传到前端
original_tokens = [] #外部变量，需要传到前端


# BertForPreTraining：
# Outputs:
#         if `masked_lm_labels` and `next_sentence_label` are not `None`:
#             Outputs the total_loss which is the sum of the masked language modeling loss and the next
#             sentence classification loss.
#         if `masked_lm_labels` or `next_sentence_label` is `None`:
#             Outputs a tuple comprising
#             - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
#             - the next sentence classification logits of shape [batch_size, 2].

# from_pretrained：
# Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
# Download and cache the pre-trained model file if needed.

# In[254]:


import re
def convert_text_to_examples(text): 
    '''功能：
            把输入的文本变成一个实例，一个实例中包含text_a,text_b(text_b用于是否为上下句的任务，该任务不使用此功能)
       输入：
            text：一个列表结构，列表中包含原始文本字符串，由于仅完成mlm任务，所以text列表中仅包含一个字符串，就是待检查的字符串
       输出：
            example：实例，其中包含：
                unique_id：此任务仅用到0
                text_a：text列表内的字符串
                text_b：此任务下该变量为None
    '''
    examples = []
    unique_id = 0
    if True:
        for line in text:
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line) #想要匹配这样的字符串'You are my sunshine. ||| I love you.'
            
            if m is None:
                text_a = line
            else:
                text_a = m.group(1) #匹配的第一句,比如You are my sunshine,my only sunshine.
                text_b = m.group(2) #匹配的第二句，比如I love you.
            
            examples.append(
                InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
    return examples
#print(convert_text_to_examples(['I love you. The cat is so cute.'])[0].text_a)

def convert_examples_to_features(examples, tokenizer, append_special_tokens=True, replace_mask=True, print_info=False):
    '''功能：
            把实例变成一个特征列表
       输入：
            examples：实例，convert_text_to_examples()函数的输出
            tokenizer：BERT的tokenizer，用于将文本进行各种处理，它可以把一个text转变成tokens，把tokens变成每个token在词典中的编号以及逆运算
            append_special_tokens：是否允许在生成的tokens中加入特殊符号，也就是[CLS]、[MASK]和[SEP]，默认为True
            replace_mask：不明
            print_info：不明
       输出：
            features：每一个feature包含：
                unique_id：编号，目前实现的功能features里面仅有一个feature
                tokens=tokens,tokens：是形如['i','love','you','.']的一个列表
                input_ids=input_ids：字符串中的每个单词在词典中的index序列
                input_mask=input_mask：一堆1
                input_type_ids=input_type_ids))：对text_a,text_b的区分，用于上下句任务，对于本任务，该参数为一个列表，其中包含token长度个的0
    '''
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a) #tokenize的作用是把"i love you."变成['i','love','you','.']
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        tokens = []
        input_type_ids = [] #segment embedding
        if append_special_tokens: #输入参数中默认为true
            tokens.append("[CLS]")
            input_type_ids.append(0)
        for token in tokens_a:
            if replace_mask and token == '_':  # XD
                token = "[MASK]"
            tokens.append(token)
            input_type_ids.append(0)
        if append_special_tokens:
            tokens.append("[SEP]")
            input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                if replace_mask and token == '_':  # XD
                    token = "[MASK]"
                tokens.append(token)
                input_type_ids.append(1)
            if append_special_tokens:
                tokens.append("[SEP]")
                input_type_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens) #把原来句子中的词语编成在字典中的编号
        input_mask = [1] * len(input_ids) 
        
        if ex_index < 5:
#             logger.info("*** Example ***")
#             logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
#             logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
#             logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
#             logger.info(
#                 "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
            
        features.append(
            InputFeatures(
                unique_id=example.unique_id,#编号，目前实现的功能features里面仅有一个feature
                tokens=tokens,#形如['i','love','you','.']的一个列表
                input_ids=input_ids,#字符串中的每个单词在词典中的index序列
                input_mask=input_mask, #一堆1
                input_type_ids=input_type_ids)) #第0类和第1类，对text_a,text_b的区分，本代码中全都是零
    return features            

def copy_and_mask_feature(feature, step, masked_tokens=None): 
    '''
        功能：
            输入feature生成训练的批次数以及mask好的训练素材
        输入：
            feature：convert_examples_to_features函数的输出
            step：两个[mask]位置的步长
            masked_tokens：默认为None，在程序中没有使用
    '''
    import copy
    tokens = feature.tokens
    len_token = len(tokens)
    if len_token<step:
        batches = range(0,len(tokens))
    else:
        batches = range(0,step)
    
    assert len_token > 0
    masked_feature_copies = []
    for i in batches: #用[mask]依次掩盖每一个位置
        feature_copy = copy.deepcopy(feature)
        masked_pos = i
        while masked_pos < len_token:
            feature_copy.input_ids[masked_pos] = tokenizer.vocab["[MASK]"]
            masked_pos = masked_pos + step
        masked_feature_copies.append(feature_copy)
    return masked_feature_copies, batches

#masked_feature_copies, batches = copy_and_mask_feature(features[0],3)
#print(masked_feature_copies[0].input_ids) #结果[101, 1045, 2293, 103, 102]
#print(batches) #结果是一个range(0,5)


# In[7]:


analyzed_cache = {}
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG
#print (lemma('gave'))
#print (lexeme('production'))
#print (conjugate(verb='give',tense=PRESENT,number=SG))
def process_text(text): 
    '''
        功能：
            处理输入文本，将文本按句子分成若干token，得出原来text中index位置的单词在x句子的y位置，还得出各个句子类别码
        输入：
            text：文本字符串，注意区别
        输出：
            input_ids_sen：二维列表，第一维列表的元素是每个句子的input_ids列表
            input_type_ids_sen：二维列表，第一维列表的元素是每个句子的input_type_ids列表
            in_sentence：通过这个二维数组可以很方便的通过在完整text中的下标找到这个下标所在的句子和在句子中的下标
            sentences：字符串列表，列表中每一个元素是一个句子字符串
            entire_ids：整个text的input_ids
            entire_type_ids：整个text的input_type_ids
    '''
    token =[]
    entire_type_ids = []
    token0 = tokenizer.tokenize(text)
    token.append('[CLS]')
    entire_type_ids.append(0)
    for i in token0:
        token.append(i)
        entire_type_ids.append(0)
    token.append('[SEP]')
    entire_type_ids.append(0)
    
    entire_ids = tokenizer.convert_tokens_to_ids(token)
    in_sentence = [[0,0]] 
    sentence_n = 0
    index = 1
    for i in range(1,len(token)-1):
        in_sentence.append([sentence_n,index])  #每个token中的词在所在句中的位置表示出来，以及该位置在哪一句中
        index = index + 1                           #比如，位置i这个词在第sentence句的index位置上
        if token[i] == '.':
            sentence_n = sentence_n + 1
            index = 1
    sentences = text.split(".")
    
    sen_token = []
    input_ids_sen = []
    input_type_ids_sen = []
    for i,sentence in enumerate(sentences):
        sentence = sentence + '.'
        sentences[i] = sentences[i] + '.'
        token = []
        input_type_ids = []
        tokens = tokenizer.tokenize(sentence)
        token.append('[CLS]')
        input_type_ids.append(0) 
        for i in tokens:
            token.append(i)
            input_type_ids.append(0)        
        token.append('[SEP]')        
        input_type_ids.append(0)
        input_ids_sen.append(tokenizer.convert_tokens_to_ids(token))
        input_type_ids_sen.append(input_type_ids)
    return input_ids_sen,input_type_ids_sen,in_sentence,sentences,entire_ids,entire_type_ids


# In[8]:


def get_word(index):
    '''
        输入：
            index：在完整text中的位置
        输出
            word:该位置上的单词
    '''
    word_id = entire_ids[index]
    word = tokenizer.ids_to_tokens[word_id]
    return word


# In[1559]:


import copy
import nltk
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG,PRESENT,SG,INFINITIVE, PRESENT, PAST, FUTURE, PROGRESSIVE

def give_suggestion(input_ids_,input_type_ids_,id_in_sen,alternative_word,threshold):
    '''
        功能：
            给出指定文本指定位置的推荐用词
        输入：
            input_ids_：要分析的文本的input_ids
            input_type_ids_：要分析的文本的的input_type_ids
            id_in_sen：要分析的文本中[MASK]的位置下标，也就是需要给出建议用词的位置
            alternative_word：推荐的备选词范围
            threshold：阈值
        输出：
            suggestion：推荐
            need：推荐的是否是备选词中的词
            suggestion_prob：推荐词填在id_in_sen位置的概率
            top_of_alternative:备选词中最值得推荐的词
    '''
    input_ids = copy.deepcopy(input_ids_)
    input_type_ids = copy.deepcopy(input_type_ids_)
    word0 = input_ids[id_in_sen]
    word0 = tokenizer.ids_to_tokens[word0]
    list_word_id = []
    
    input_ids[id_in_sen] = tokenizer.vocab["[MASK]"]
    T_input_ids = torch.tensor([input_ids], dtype=torch.long) #把input_ids增加了一个维度
    T_input_type_ids = torch.tensor([input_type_ids], dtype=torch.long) #把input_type_ids增加了一个维度，其实每一行都一样
    T_input_ids = T_input_ids.to(device) #拿去GPU
    T_input_type_ids = T_input_type_ids.to(device)

    mlm_logits = model(T_input_ids)
    mlm_probs = F.softmax(mlm_logits, dim=-1)
    reduced_mlm_probs = mlm_probs[0][id_in_sen]

    top_ind = reduced_mlm_probs.argmax().item()
    top_prob = reduced_mlm_probs.max().item() 
    
    list_word = []
    
    top_of_alternative = None
    if len(alternative_word)>0:
        list_word_prob = {}
        for word in alternative_word:
            try:
                list_word_id.append(tokenizer.vocab[word])
                list_word.append(word)
            except KeyError:
                pass

        for word,word_id in zip(list_word,list_word_id):
            list_word_prob.update({word:float(reduced_mlm_probs[word_id].data)})
        prob_ord = sorted(list_word_prob.items(),key = lambda x:x[1],reverse = True)
        
        top_prob_word = prob_ord[0][1]
        top_of_alternative = prob_ord[0][0]
        gap = math.log(top_prob) - math.log(top_prob_word)
        
        if gap < threshold:
            suggestion = prob_ord[0][0]
            suggestion_prob = prob_ord[0][1]
            need = 1
        else:
            suggestion = tokenizer.ids_to_tokens[top_ind]
            suggestion_prob = top_prob
            need = 0
        #print("gap = " + str(gap))
        #print(prob_ord)
    else:
        suggestion = tokenizer.ids_to_tokens[top_ind]
        suggestion_prob = top_prob
        need = 0
        
    return suggestion,need,suggestion_prob,top_of_alternative 

#返回变量5
#suggestion -> 最值得推荐的词
#need -> 是否需要可选词中的一个
#suggestion_prob ->最值得推荐的词的概率
#top_of_alternative -> 可选词中最值得推荐的
#suggestion,need,suggestion_prob,top_of_alternative = give_suggestion(input_ids_,input_type_ids_,id_in_sen,alternative_word,threshold)


# In[1473]:


from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
from pattern.en import comparative, superlative
from pattern.en import suggest
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
import enchant
d = enchant.Dict("en_US")


# In[1474]:


stemmers=[]
stemmers.append(LancasterStemmer()) 
stemmers.append(SnowballStemmer("english"))
stemmers.append(PorterStemmer())
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
def word_convert(word,new_word,Stemmer):
    '''
        功能：
            根据提供的word和可能的变形new_word,得到正确的变形，例如给出basic，basicly得到basically
        输入：
            word：需要变形的词
            new_word:猜想的变形
        输出：
            suggest_word:推荐的正确变形
    '''
    suggest_word = None
    word_stem = Stemmer().stem(word)
    suggest_ = new_word
    
    suggest_list = suggest(suggest_)

    if len(word)<len(new_word):
        flag = 0
    else:
        flag = 1
    word_stem = word_stem[:-1]
    suggestion_word_stem = Stemmer().stem(suggest_)
    
    for word_ in suggest_list:
        if word == word_[0]:
            continue
        if (word_[0] == new_word and word_[1] > 0.95):# or word_[1] > 0.95 :
            suggest_word = word_[0]
            break           
        if word_[1] < 0.001:
            break
        stem_list = []
        for stemmer in stemmers:
            suggest_stem = stemmer.stem(word_[0])
            if flag == 1 and suggest_stem[:-1] in word_stem and word_stem[:3] in suggest_stem[:3]: #一般是去后缀
                suggest_word = word_[0]
                break
            elif flag == 0 and word_stem in suggest_stem and word_[0][-1:] in suggest_[-1:]: #一般是加后缀，后缀一定要一样
                suggest_word = word_[0]
                break
                
        if suggest_word != None:
            break
    return suggest_word 


# In[1475]:


stemmers=[]
stemmers.append(LancasterStemmer()) 
stemmers.append(SnowballStemmer("english"))
stemmers.append(PorterStemmer())
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
def word_convert(word,new_word,Stemmer):
    '''
        说明;
            与上面的区别是使用的拼写改错算法不同，上面那个平均速度慢，但更符合我的要求，这个平均速度更快
        功能：
            根据提供的word和可能的变形new_word,得到正确的变形，例如给出basic，basicly得到basically
        输入：
            word：需要变形的词
            new_word:猜想的变形
            Stemmer:词根提取器
        输出：
            suggest_word:推荐的正确变形
    '''
    if d.check(new_word)==True: #如果发现new_word拼写正确，则直接返回
        return new_word
    else:
        suggest_word = None
        word_stem = Stemmer().stem(word)
        suggest_ = new_word
        suggest_list = d.suggest(suggest_) #可能的正确单词列表

        if len(word)<len(new_word): #一般都是加后缀
            flag = 0
        else: #一般都是去后缀
            flag = 1
        word_stem = word_stem[:-1] #这样效果更好一点，防止某些去e加后缀或者y变i的变形被忽略
        suggestion_word_stem = Stemmer().stem(suggest_)
        for word_ in suggest_list:
            if word == word_: #如果变形和原型一样，就跳过这个词
                continue
            if (word_ == new_word): #如果推荐的和new_word一样，直接把该词作为结果
                suggest_word = word_
                break
            if ' ' in word_ or '-' in word_: #enchant.Dict模型特有的问题，一个拼写错误的词可能会给你返回一个带连字符词的或者是两个词
                continue
            stem_list = []
            for stemmer in stemmers:
                suggest_stem = stemmer.stem(word_)
                if flag == 1 and suggest_stem in word_stem and word_stem[:3] in suggest_stem[:3]: #一般是去后缀
                    suggest_word = word_
                    break
                elif flag == 0 and word_stem in suggest_stem and word_[-1:] in suggest_[-1:]: #一般是加后缀，后缀一定要一样
                    suggest_word = word_
                    break

            if suggest_word != None:
                break
        return suggest_word 


# In[1476]:


'''下面是词性转换系列函数
    功能：
        词性转变系列函数
    输入：
        word：原形词
    输出：
        suggest_word：推荐的变形
        suggest_list：推荐的变形列表
    说明：
        词性变化的能力有限，对于有些特殊变形，比如die->death，success->succeed无能为力'''


# In[1477]:



def adj_to_adv(word):
    suggest_word = None
    if(word == "good"):
        return "well"
    else:
        suggest_ = word + 'ly'
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        return suggest_word
#如果形容词副词同形，那么他会返回none，但是不影响计算，因为形容词副词同形啊


def adv_to_adj(word):
    suggest_word = None
    if(word == "well"):
        return "good"    
    elif word[-2:] == 'ly':
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
    return suggest_word



# In[1550]:


def adj_to_anything(word):#形容词变成其他词性
    suggest_word = None
    suggest_list = []
    if word[-1:] == 'y': #举例 healthy->health
        suggest_ = word[:-1]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-3:] == 'ful':#举例 successful->success
        suggest_ = word[:-3]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-3:] == 'ive': #举例 active -> act
        suggest_ = word[:-4]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-2:] == 'ed': #举例 interested->interest->interesting
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)     
        suggest_ = suggest_ + 'ing'
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)      
            
    elif word[-3:] == 'ing':#举例 interesting->interest->interested
        suggest_ = word[:-3]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
        suggest_ = suggest_ + 'ed'
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)  
            
    elif word[-4:] == 'less': #举例 careless -> care
        suggest_ = word[:-4]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-2:] == 'ly':  #举例： friendly -> friend , lovely -> love
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
 
    elif word[-1:] == 't': #举例 different -> different
        suggest_ = word[:-1]
        suggest_ = suggest_ + 'ce'
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-3:] == 'ous': #举例 dangerous -> danger
        suggest_ = word[:-3]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-2:] == 'al': #举例 original -> origin
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-4:] == 'able':
        suggest_ = word[:-4]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-2:] == 'en': #举例 woolen -> wool
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-2:] == 'ic': 
        suggest_ = word + 'al'
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)  
        suggest_ = word[:-2]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)   
    elif word[-3:] == 'ish':
        suggest_ = word[:-3]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word == None:
            suggest_ = word[:-3]
            suggest_ = suggest_ + 'and'
            suggest_word = word_convert(word,suggest_,PorterStemmer) 
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-3:] == 'ese':
        suggest_ = word[:-3]
        suggest_ = suggest_ + 'a'
        suggest_word = word_convert(word,suggest_,PorterStemmer)  
        if suggest_word != None:
            suggest_list.append(suggest_word)
    elif word[-3:] == 'ian':
        suggest_ = word[:-1]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word == None:
            suggest_ = word[:-3]
            suggest_ = suggest_ + 'y'
            suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    if suggest_word == None:
        HouZhui_list = ['ment','ness','tion','ture','sion','ty','y','tive','sive']
        for HouZhui in HouZhui_list:
            suggest_ = word + HouZhui
            new_word = word_convert(word,suggest_,PorterStemmer)
            if new_word != None:
                suggest_word = new_word
                suggest_list.append(suggest_word)
    suggest_list = list(set(suggest_list))      
    return suggest_list




# In[1551]:


def N_to_anything(word):#名词变成其他词性
    suggest_list = []
    list_HouZhui = ['y','ful','tive','sive','ed','ing','less','ly','ous','al','able','en','tic','ish','ance','er','or']
    list_QianZhui = ['a']
    if word[-4:] in ['ment','ness','tion','ture','sion','tive','sive']:
        suggest_ = word[:-4]
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    else:
        for HouZhui in list_HouZhui:
            suggest_ = word + HouZhui
            suggest_word = word_convert(word,suggest_,PorterStemmer)
            if suggest_word != None:
                suggest_list.append(suggest_word)
        for QianZhui in list_QianZhui:
            suggest_ = QianZhui + word
            suggest_word = word_convert(word,suggest_,PorterStemmer)
            if suggest_word != None:
                suggest_list.append(suggest_word)
        if word[-2:] == 'ce':
            suggest_ = word[:-2]
            suggest_ = suggest_ + 't'
            suggest_word = word_convert(word,suggest_,PorterStemmer)
            if suggest_word != None:
                suggest_list.append(suggest_word)        
        elif word[-4:] == 'land':
            suggest_ = word[:-4]
            suggest_word = word_convert(word,suggest_,PorterStemmer)
            if suggest_word == None:
                suggest_ = suggest_ + 'lish'
                suggest_word = word_convert(word,suggest_,PorterStemmer)
            if suggest_word != None:
                suggest_list.append(suggest_word)  
        #print(suggest_list)
    suggest_list = list(set(suggest_list))
    return suggest_list


# In[1552]:


def V_to_anything(word):#动词变成其他词性
    suggest_word = None
    suggest_list = []

    HouZhui_list = ['ful','tive','sive','ed','less','ly','ous','al','able','en','tic','ish','ance','tion','sion','ment','er','or','ee']
    for HouZhui in HouZhui_list:
        suggest_ = word + HouZhui
        suggest_word = word_convert(word,suggest_,PorterStemmer)
        if suggest_word != None:
            suggest_list.append(suggest_word)
    suggest_list = list(set(suggest_list))
    return suggest_list


# In[1553]:


'''
    功能：
        生成形容词，副词关联词表
    输入：
        word：形容词/副词
    输出：
        list_word：为没有添加词的其他形式，包括三音节以下词的比较级最高级
        list_word2：为三音节及以上的词的比较级最高级，如果输入形容词比较级最高级没有more/most，该列表为空
    说明：
        由于三音节形容词/副词的比较级，最高级为more/most+原形容词/副词，所以特别把形容词/副词和其他词性变形区分出来
'''

def build_like_word_adj(word): #创建类似形容词列表
    list_word = []
    list_word2 = [] #把比较级最高级带more的放在这里
    lemmas = lemmatizer(word, u'adj')
    #print(lemmas)
    for i in lemmas:
        list_word.append(i)
        word_er = comparative(i)
        if "more" in word_er:  #把比较级带more，most的词放在另一个列表list_word2
            list_word2.append(word_er)
        else:
            list_word.append(word_er)
        word_est = superlative(i)
        if "most" in word_est:
            list_word2.append(word_est)
        else:
            list_word.append(word_est)
        word_adv = adj_to_adv(i)
        if word_adv != None:
            list_word.append(word_adv)
    list_N = adj_to_anything(word)
    for N in list_N:
        list_word.append(N)
        
    list_word = list(set(list_word))
    return list_word,list_word2

def build_like_word_adv(word): #创建类似形容词列表
    list_word = []
    list_word2 = []
    list_special = ['however','seldom','often','never','otherwise']
    if word in list_special:
        list_word = [word]
        list_word2 = []
    else:
        lemmas = lemmatizer(word, u'adj')
        #print(lemmas)
        for i in lemmas:
            list_word.append(i)
            word_er = comparative(i)
            if "more" in word_er:
                list_word2.append(word_er)
            else:
                list_word.append(word_er)
            word_est = superlative(i)
            if "most" in word_est:
                list_word2.append(word_est)
            else:
                list_word.append(word_est)
            word_adv = adv_to_adj(i)
            if word_adv != None:
                list_word.append(word_adv)
    list_word = list(set(list_word))
    return list_word,list_word2


# In[1554]:


'''
    功能：
        根据检查的位置整理出放入BERT模型的input_ids,input_type_ids以及检查位置在input_ids中的下标位置
        pre_training_input_in_sentence得到检查位置所在句子的信息
        pre_training_input_entire得到检查位置在完整text中的信息
    输入：
        index：在完整text中的位置
    输出：
        word：该下标下的单词
        input_ids：tokens的对应字典id列表
        input_type_ids：零列表
        id_in_sen：检查位置在句子中的下标(pre_training_input_in_sentence的返回)
        index：检查位置在完整text中的下标，其实就是输入的下标
'''
def pre_training_input_in_sentence(index): 
    sentence_id = in_sentence[index][0]
    id_in_sen = in_sentence[index][1]
    word = input_ids_sen[sentence_id][id_in_sen]
    word = tokenizer.ids_to_tokens[word]
    input_ids = copy.deepcopy(input_ids_sen[sentence_id])
    input_type_ids = copy.deepcopy(input_type_ids_sen[sentence_id])

    return word,input_ids,input_type_ids,id_in_sen

def pre_training_input_entire(index): 
    word = entire_ids[index]
    word = tokenizer.ids_to_tokens[word]
    input_ids = copy.deepcopy(entire_ids)
    input_type_ids = copy.deepcopy(entire_type_ids)

    return word,input_ids,input_type_ids,index

#[101, 1045, 2572, 3153, 2006, 1996, 2754, 1012, 102]
#[101, 1045, 2572, 3153, 2006, 1996, 2754, 1012, 1045, 2018, 1037, 2200, 2204, 2835, 1012, 1996, 2377, 2001, 2200, 5875, 1012, 102]


# In[1555]:


import math
from pattern import en
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG,INFINITIVE, PRESENT, PAST, FUTURE, PROGRESSIVE


'''
    功能：
        1.judge_and_suggestion系列函数，这个系列函数是在analyse之前做的一个预先判断处理，判断的是该位置原来词的相关词中有没有可以代替它的词
        2.当相关词中有词的可能性和原词的可能性的差距大于阈值，则认为原词是错的，可以用相关词替换
        3.替换词的gap还要经过后续的检查才能决定他是不是最好的推荐，这一步骤放在了show_abnormals里
    输入：
        prob：该位置可能性列表
        original：该位置原先的词
        list_word：该位置相关词表
        threhold：门槛，也就是阈值
    输出：
        judge：判断原来的词是否正确，0表示需要换词，1表示不需要换词或者说相关词里面没一个合适的
        suggestion：相关词中最好的推荐
        gap_with_totally_top:备选词中概率最高的和所有词中概率最高的之间的gap,可以换的词也有可能因为gap太大而遭到拒绝
'''
def judge_and_suggestion(prob,original,list_word,threhold):
    top_prob = 0
    list_word = list_word + [original]
    original_prob = prob[tokenizer.vocab[original]]
    best = None
    suggestion = None
    for word in list_word:
        try:
            word_id = tokenizer.vocab[word]
            prob_word = prob[word_id]
            if prob_word > top_prob:
                top_prob = prob_word
                best_word = word
        except KeyError:#有的词enchant认为是正确的拼写，bert的词典里却没有，比如tiring，这种情况暂时没法解决，但是实际上bert不认的词会自动分词
            pass

    totally_top = prob.max().item() #最高的概率（不需要知道概率最大的词是哪一个）
    gap_with_origin = math.log(top_prob) - math.log(original_prob) #备选词中最大概率和原来的词的概率的差
    gap_with_totally_top = math.log(totally_top) - math.log(top_prob) #所有词中最高的概率和备选词中最高的概率的差
    
    if gap_with_origin > threhold:
        suggestion = best_word
        return 0,suggestion,gap_with_totally_top
    else:
        return 1,suggestion,gap_with_totally_top
    


# In[1556]:


'''分析各种词性系列函数
    功能：对第一遍检查得出的有问题的位置的单词，根据不同的词性进行不同步骤的分析
    输入：
        index：在原文中的错误位置
        prob：该位置可能性列表
        gap：原文该位置的词和概率最高的词之间的gap
        top_word：概率最高的词
        threshold:免检查门槛
        threshold2:免修正门槛(勉强不算错)
        threshold3:用推荐词替换的最低要求，大于该阈值才可以替换
    输出：
        suggestion:给出的修改建议，修改建议不局限于错误位置
    说明：
        不仅局限于错误位置的分析是通过预添加或者去掉一个token，多进行一次model计算
'''


# In[1557]:


import copy
import nltk
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG,PRESENT,SG,INFINITIVE, PRESENT, PAST, FUTURE, PROGRESSIVE

def analyse_V(index,prob,gap,top_word,threshold,threshold2,threshold3):
#这是一个处理动词语法问题的函数，输入为问题词在text的token中的下标index
    if gap < threshold:
        return None
    #******************************top_word暗示我应该是不定式**************************
    if top_word in ["to","for"]:
        wordV,input_ids,input_type_ids,index = pre_training_input_entire(index)
        input_ids.insert(index,tokenizer.vocab['to'])
        input_type_ids.append(0)
        list_word = [conjugate(verb=wordV,tense=PRESENT,person = 1)]
        suggestion,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_word,5) 
        if need == 1:
            return 'to ' + suggestion 
        
    #*****************************判断是不是时态或者拼写错误，又或者是其他词性********
    wordV = get_word(index)
    #这三种是不涉及位置变化的检查，根据生成词表的速度从快到慢依次检查，之后也不需要再生成词表

    list_V = lexeme(wordV)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordV,list_V,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion
    
    list_others = V_to_anything(conjugate(verb=wordV,tense=PRESENT,person = 1))
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordV,list_others,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion    
        
    list_spell_correct = d.suggest(wordV)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordV,list_spell_correct,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion

    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    
    front_word = get_word(index - 1)
    behind_word = get_word(index + 1)
    #**************************************判断是不是缺介词***************************
    list_IN = ["to","at","in","on","by","for","from","with","about","against","along","among","around","as","before","behind","below","beside","between","during","besides","into","near","over","through","under","without","after","above","of"]
    if behind_word not in list_IN:
        print("检查点")
        wordV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index)
        input_ids.insert(id_in_sen + 1,tokenizer.vocab['at'])#就随便插入一个东西，占位子
        input_type_ids.append(0)
        suggestion_IN,need_IN,_,_ = give_suggestion(input_ids,input_type_ids,id_in_sen + 1,list_IN,2)
        if need_IN == 1:
            input_ids[id_in_sen + 1] = tokenizer.vocab[suggestion_IN]
            list_word = list_V
            suggestion_V,need,_,_ = give_suggestion(input_ids,input_type_ids,id_in_sen,list_word,5)
            if need == 1:
                suggestion = suggestion_V + ' ' + suggestion_IN
                return suggestion
    
    need_to_will = need_be = 0
    
    #**************************************判断是不是不定式或者将来时***************************  
    if front_word not in ["to","will"]:
        wordV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index)
        input_ids.insert(id_in_sen,tokenizer.vocab['to'])#就随便插入一个东西，占位子
        input_type_ids.append(0)
        try:
            input_ids[id_in_sen + 1] = tokenizer.vocab[conjugate(verb=wordV,tense=PRESENT,person = 1)]
            suggestion_to_will,need_to_will,prob0,_ = give_suggestion(input_ids,input_type_ids,id_in_sen,["to","will"],1)
        except KeyError:
            need_to_will = 0
    #**********************************判断是不是被动语态或者进行时*******************   
    list_be = lexeme('be')
    list_be = lexeme('be')[:8] #把否定去掉  
    #********************是不是被动语态****************   

    wordV,input_ids,input_type_ids,index = pre_training_input_entire(index)
    input_ids.insert(index,tokenizer.vocab['be'])#就随便插入一个东西，占位子
    input_type_ids.append(0)
    try:
        input_ids[index + 1]=tokenizer.vocab[conjugate(verb=wordV,tense=PAST,aspect=PROGRESSIVE)]
        suggestion1,need_be1,prob1,_ = give_suggestion(input_ids,input_type_ids,index,list_be,1)
    except KeyError:
        need_be1 = 0
        
    #********************是不是现在分词****************   
    try:
        input_ids[index + 1]=tokenizer.vocab[conjugate(verb=wordV,tense=PRESENT,aspect=PROGRESSIVE)]
        suggestion2,need_be2,prob2,_ = give_suggestion(input_ids,input_type_ids,index,list_be,1)
        #print(tokenizer.convert_ids_to_tokens(input_ids))
    except KeyError:
        need_be2 = 0

    #***************************选择是不定式还是被动语态还是进行时****************************
    prob_max = 0
    if need_to_will == 1:
        prob_max = max(prob_max,prob0)
    if need_be1 == 1:
        prob_max = max(prob_max,prob1)
    if need_be2 == 1:
        prob_max = max(prob_max,prob2)

    if need_to_will == 1 and prob_max == prob0:
        need_be = 0
    if need_be1 == 1 and prob_max == prob1:
        need_to_will = 0
        need_be = 1
        be_ = suggestion1
    if need_be2 == 1 and prob_max == prob2:
        need_to_will = 0
        need_be = 1
        be_ = suggestion2
    #*************************************************处理各种语法******************************************************************
    if need_to_will == 1:
        wordV,input_ids,input_type_ids,index = pre_training_input_entire(index)
        input_ids.insert(index,tokenizer.vocab[suggestion_to_will])
        input_type_ids.append(0)
        list_word = [conjugate(verb=wordV,tense=PRESENT,person = 1),conjugate(verb=wordV,tense=PRESENT,aspect=PROGRESSIVE)]
        suggestion,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_word,5)
        if need == 1:
            return 'to ' + suggestion
        else:
            return top_word

    elif need_be == 1:
        #********************************被动语态或者进行时*****************
        wordV,input_ids,input_type_ids,index = pre_training_input_entire(index)
        input_ids.insert(index,tokenizer.vocab[be_])
        input_type_ids.append(0)
        list_word = lexeme(wordV)
        suggestion,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_word,5)
        if need == 1:
            return be_ + ' '+ suggestion
        else:
            return top_word
    else:
        return top_word
    
    return suggestion
    


# In[1558]:


def analyse_adj(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None
    wordADJ = get_word(index)
    #*****************************判断是不是时态或者拼写错误，又或者是其他词性********
    
    list_word,list_word2 = build_like_word_adj(wordADJ)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordADJ,list_word,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    list_spell_correct = d.suggest(wordADJ)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordADJ,list_spell_correct,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion

    #list_word = list_word + list_spell_correct
    front_word = get_word(index - 1)
    behind_word = get_word(index + 1)
    if front_word in ['more','most'] and len(list_word2) == 0:
        #判断是不是比较级使用错误,如果该形容词比较级/最高级不需要加more/most，但是前面有more/most
        wordADJ,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        del input_ids[id_in_sen - 1]
        del input_type_ids[0]
        suggestion3,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen - 1,list_word,min(threshold2, gap - threshold3))
        return '去掉前面 ' + get_word(index - 1)+ ' 原位置改成 ' + suggestion3
    
    elif behind_word in ['##er','##r'] and len(list_word2) != 0:
        #判断是不是比较级使用错误,如果该形容词比较级/最高级需要more/most，但是错写成形容词+er/est
        wordADJ,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        input_ids[id_in_sen] = tokenizer.vocab['more']
        suggestion5,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen + 1,list_word,min(threshold2, gap - threshold3))
        return '去掉后面 '+ get_word(index + 1) + ' 原位置改成 '+ 'more' + ' ' + suggestion5  
    
    elif behind_word in ['##est','##st'] and len(list_word2) != 0:
        #判断是不是比较级使用错误,如果该形容词比较级/最高级需要more/most，但是错写成形容词+er/est
        wordADJ,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        input_ids[id_in_sen] = tokenizer.vocab['most']
        suggestion5,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen + 1,list_word,min(threshold2, gap - threshold3))
        return '去掉后面 '+ get_word(index + 1) + ' 原位置改成 '+ 'most' + ' ' + suggestion5  
    
        
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    
    if front_word not in ['this','that','these','those','more','most']:#检查形容词前面是否需要加冠词或者是需要more，most的比较级，最高级抑或是be动词
        wordADJ,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        input_ids.insert(id_in_sen,tokenizer.vocab["[MASK]"])
        input_type_ids.append(0)
        list_front = ['the','a','an','this','that','these','those','some','any','all','more','most','am','is','are','was','were'] 
        suggestion,need_front,_,_= give_suggestion(input_ids,input_type_ids,id_in_sen,list_front,2)
        if need_front == 1:
            wordADJ,input_ids,input_type_ids,index = pre_training_input_entire(index)
            input_ids.insert(index,tokenizer.vocab[suggestion])
            input_type_ids.append(0)
            suggestion2,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_word,min(threshold2, gap - threshold3))     
            if need == 1:
                return suggestion + ' ' + suggestion2
            else:
                return top_word
        
    return top_word


# In[1600]:


def analyse_adv(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None
    
    wordADV = get_word(index)
    if wordADV in ['not']:
        return None
    #*****************************判断是不是时态或者拼写错误，又或者是其他词性********
    
    list_word,list_word2 = build_like_word_adv(wordADV)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordADV,list_word,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    list_spell_correct = d.suggest(wordADV)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordADV,list_spell_correct,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion

    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    
    #list_word = list_word + list_spell_correct
    if get_word(index - 1) in ['more','most'] and len(list_word2) == 0:
        #判断是不是比较级使用错误,这个if语句处理：该形容词比较级/最高级不需要加more/most，但是前面有more/most 
        wordADV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        del input_ids[id_in_sen - 1]
        del input_type_ids[0]
        suggestion3,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen - 1,list_word,5)
        return '去掉前面 ' + get_word(index - 1)+ ' 原位置改成 ' + suggestion3
    
    elif get_word(index + 1) in ['##er','##r'] and len(list_word2) != 0:
        #判断是不是比较级使用错误,如果该形容词比较级/最高级需要more/most，但是错写成形容词+er/est
        wordADV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        input_ids[id_in_sen] = tokenizer.vocab['more']
        suggestion5,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen+1,list_word,5)
        return '去掉后面 '+ get_word(index + 1) + ' 原位置改成 '+ 'more' + ' ' + suggestion5  
    
    elif get_word(index + 1) in ['##est','##st'] and len(list_word2) != 0:
        #判断是不是比较级使用错误,如果该形容词比较级/最高级需要more/most，但是错写成形容词+er/est
        wordADV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index) 
        input_ids[id_in_sen] = tokenizer.vocab['most']
        suggestion5,need_adj,prob,_ = give_suggestion(input_ids,input_type_ids,id_in_sen+1,list_word,5)
        return '去掉后面 '+ get_word(index + 1) + ' 原位置改成 '+ 'most' + ' ' + suggestion5  

    else:
        #检查形容词前面是否需要加冠词或者是需要more，most的比较级，最高级，be动词
        wordADV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index)
        input_ids.insert(id_in_sen,tokenizer.vocab["[MASK]"])
        input_type_ids.append(0)
        list_front = ['the','a','an','this','that','these','those','some','any','all','more','most','am','is','are','was','were'] 
        suggestion,need_front,_,_= give_suggestion(input_ids,input_type_ids,id_in_sen,list_front,2)
        if need_front == 1:
            wordADV,input_ids,input_type_ids,index = pre_training_input_entire(index)
            input_ids.insert(index,tokenizer.vocab[suggestion])
            input_type_ids.append(0)
            #print(tokenizer.convert_ids_to_tokens(input_ids))
            suggestion2,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_word,5)   
            if need == 1:
                return suggestion + ' ' + suggestion2
            else:
                return top_word
        else:
            wordADV,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index)
            input_ids.insert(id_in_sen + 1,tokenizer.vocab[","])
            input_type_ids.append(0)
            suggestion3,need_douhao,_,_= give_suggestion(input_ids,input_type_ids,id_in_sen,list_word,2)
            if need_douhao == 1:
                return suggestion3 + ' ,'
            else:
                return top_word


# In[1536]:


from pattern.en import article,referenced,pluralize, singularize
import nltk
def analyse_N(index,prob,gap,top_word,threshold,threshold2,threshold3):
    #这是一个处理名词语法问题的函数，输入为问题词在text的token中的下标index
    if gap < threshold:
        return None
    
    wordN = get_word(index)
    #*****************************判断是不是时态或者拼写错误，又或者是其他词性********
    word_tag = nltk.pos_tag([wordN])
    if word_tag[0][1] == "NN":
        N_ = wordN
        N_s= pluralize(wordN)
    else:
        N_ = singularize(wordN)
        N_s= wordN
    list_N = [N_,N_s]
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordN,list_N,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion
    
    list_others = N_to_anything(N_)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordN,list_others,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
        
    list_spell_correct = d.suggest(wordN)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordN,list_spell_correct,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion

    #***********************************************************************************************************************************
    need_DT = 0 #表示是否需要在前面加冠词 
    wordN,input_ids,input_type_ids,id_in_sen = pre_training_input_in_sentence(index)

    #*****************************************判断是否需要冠词或介词************************************************************************   
    list_DT = ['the','a','an']
    front_word = get_word(index - 1)
    if front_word in list_DT:#如果前一个词就是冠词，那么一定不需要再往前面加介词或冠词
        if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
            return None
        else:
            return top_word
    
    input_ids.insert(id_in_sen,tokenizer.vocab["[MASK]"])
    input_type_ids.append(0)
    list_IN = ["of",'to',"at","in","on","by","for","from","with","about","against","along","among","around","as","before","behind","below","beside","between","during","besides","into","near","over","through","under","without","after","above"]
    list_DT_IN = list_DT + list_IN
    suggestion,need_DT_IN,_,_= give_suggestion(input_ids,input_type_ids,id_in_sen,list_DT_IN,2)
    if need_DT_IN == 0:#不需要冠词或介词
        if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
            return None
        else:
            return top_word
        
    elif need_DT_IN == 1:#需要冠词或介词
        wordN,input_ids,input_type_ids,index = pre_training_input_entire(index)
        input_ids.insert(index,tokenizer.vocab[suggestion])
        input_type_ids.append(0)
        suggestion2,need,_,_= give_suggestion(input_ids,input_type_ids,index + 1,list_N ,min(9.5,gap - threshold3))
        if need == 1:
            return suggestion + ' ' + suggestion2
        
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    else:
        return top_word


# In[1537]:


'''
    这是一个相关代词的词典，容易混淆的词放在一个列表中

'''
like_he = ['he','his','him','himself','who', 'whom', 'whose']
like_she = ['she','her','herself','hers','who', 'whom', 'whose']
like_it = ['it','its','itself','who', 'whom', 'whose']
like_i = ['i','me','my','myself','mine']
like_you = ['you','your','yourself','yourselves']
like_we = ['we','us','our','ours','ourselves']
like_they = ['they','them','their','theirs']

like_this = ['this', 'these'] 
like_that = ['that','those'] 
pronoun_Question = ['who', 'whom', 'whose', 'which', 'what', 'whoever', 'whichever', 'whatever'] #疑问代词
pronoun_relation =  ['that', 'which', 'who', 'whom', 'whose', 'as'] #关系代词
like_some = ['some','any']
like_few = ['few','little']
like_many = ['many','much']
like_other = ['another','other']

pronoun = [like_he,like_she,like_it,like_i,like_you,like_we,like_they,like_this,like_that,pronoun_Question,pronoun_relation,like_some,like_few,like_many,like_other]
pronoun_dictionary = {}
pronoun_list = []
for list_word in pronoun:
    pronoun_list = pronoun_list + list_word
    for word in list_word:
        pronoun_dictionary.update({word:list_word})


# In[1538]:


def analyse_pronoun(index,prob,gap,top_word,threshold,threshold2,threshold3):
    #这是一个处理代词语法问题的函数，输入为问题词在text的token中的下标index
    if gap < threshold:
        return None
    
    wordPROP = get_word(index)
    #*****************************判断是不是时态或者拼写错误，又或者是其他代词********
    try:
        list_PROP = pronoun_dictionary[wordPROP]
    except:
        list_PROP = []
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordPROP,list_PROP,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  

    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    else:
        judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordPROP,pronoun_list,threshold3)#在所有代词里面选择
        if judge==0 and gap_with_totally_top < threshold2:
            return suggestion 
        else:
            return None


# In[1613]:


def analyse_DT(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None  
    
    wordDT = get_word(index)
    if wordDT in ["every",'per','each','no']:#有实际意义，不做修改
        return None

    if wordDT in ['some']:
        list_word = ['some','any','a','an']
    elif wordDT in ['any']:
        list_word = ['some','any',"every",'per','each']
    elif wordDT in ['this','that','these','those']:
        list_word = ['this','that','these','those']
    elif wordDT in ['the','a','an']:
        list_word = ['the','a','an','some','any']
    elif wordDT in ['another','other']:
        list_word = ['another','other']
    elif wordDT in ['all','both']:
        list_word = ['all','both']
    else:
        list_word = [wordDT]
    
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordDT,list_word,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    
    elif top_word in ["at","in","on","by","for","from","with","about","against","along","among","around","as","before","behind","below","beside","between","during","besides","into","near","over","through","under","without","after","above","of",'to']:
        return top_word + ' ' + wordDT
    else:
        if top_word in ['some','any','this','that','these','those','the','a','an']:
            return top_word
        elif wordDT in ['another','other','all','both']:
            return None
        else:
            return "去掉 " + wordDT
# In[1614]:


def analyse_IN(index,prob,gap,top_word,threshold,threshold2,threshold3):
    #检查介词，确认需不需要删掉或者换介词
    if gap < threshold:
        return None  
    
    wordIN = get_word(index)
    if wordIN in ['before',"after","above","below","underneath","beneath","without"]:#有实际意义，不做修改
        return None
    
    list_word = ["at","in","on","by","for","from","with","about","against","along","among","around","as","before","behind","below","beside","between","during","besides","into","near","over","through","under","without","after","above","of",'to']
    
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordIN,list_word,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    list_spell_correct = d.suggest(wordIN)
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordIN,list_spell_correct,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion    
    
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    elif top_word in u',.!?[]()<>"\'':
        return top_word
    else:
        return "去掉 " + wordIN
#print(analyse_IN(76))


# In[1615]:


def analyse_CC(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None  
    
    wordCC = get_word(index)
    list_CC = ["but","because","yet","still","however","although","so","thus","and","or","too","either","or","neither","nor","when","while","as","whenever","since","until","till",","]
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordCC,list_CC,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    else:
        return None


# In[1616]:


def analyse_MD(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None     
    
    wordMD = get_word(index)
    if wordMD in ['can','could']:
        list_MD = ['can','could']
    elif wordMD in ['may','might']:
        list_MD = ['may','might']
    elif wordMD in ['shall','should']:
        list_MD = ['shall','should']   
    elif wordMD in ['will','would']:
        list_MD = ['will','would']  
    elif wordMD in ['dare','dared']:
        list_MD = ['dare','dared']  
    else:
        list_MD = [wordMD]
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,wordMD,list_MD,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    else:
        return None


# In[1617]:


def analyse_biaodian(index,prob,gap,top_word,threshold,threshold2,threshold3):
    if gap < threshold:
        return None     
    
    biaodian = get_word(index) 
    biaodian_list = ['.',',',';','!','?','"',"'",'，','。','’','‘','“','”','and','but']
    judge,suggestion,gap_with_totally_top = judge_and_suggestion(prob,biaodian,biaodian_list,threshold3)
    if judge==0 and gap_with_totally_top < threshold2:
        return suggestion  
    
    if gap < threshold2:#没有可以替换的词，而且原本该位置的词就勉强符合要求
        return None
    else:
        return None


# In[1618]:


'''
    功能：
        这是几个和拼写检查相关函数
        correct_spelling：用于发现text中拼写错误，写成不存在的词的情况，并暂时把它改成存在的词，这样再放入模型训练，完成之后的步骤
        token_Align：展示拼写错误时需要将原来错误的词显示出来，由于BERT的tokenize会把错误的词分段，造成未知序号的混乱，因而需要将原来的token和被correct的token位置对齐
        这两个函数需要配合使用
'''
import enchant
import re
d = enchant.Dict("en_US")
from pattern.en import suggest

def C_trans_to_E(string): #标点符号转换函数
    E_pun = u',.!?[]()<>"\'"\'.:;'
    C_pun = u'，。！？【】（）《》“‘”’．：'
    table= {ord(f):ord(t) for f,t in zip(C_pun,E_pun)}
    return string.translate(table)

def process_biaodian(text):#把标点和字母分开，使得用split分词能把标点分成单独的token,顺便把中文标点变成英文标点
    text1 = ''
    for character in text[0]: 
        if character in u',.!?[]()<>"\':-;，。！？【】（）《》“‘”’．%':
            character1 = C_trans_to_E(character)
            text1 = text1 + ' '+character1+' '
        else:
            text1 = text1 + character 
    return [text1]

def correct_spelling(text):
    #text:原本可能带有拼写错误的文本
    #返回[correct_text]：不带拼写错误的文本,外面套上中括号，保持列表的形式
    global suggestions
    correct_text = ''
    text0 = text
    text1 = ''
    
    tokens = text.split(' ')
    for token in tokens: #给拼写错误的单词标上‘错’
        if token not in ['.',',',';','!','?','"',"'",'，','。','’','‘','“','”',"\r\n",""]:
            if d.check(token)==False and token != suggest(token)[0][0]:
                word =  '不' + suggest(token)[0][0] #pattern的suggestion 
            else:
                word = token
        elif token == "\r\n":
            word = '换'
        else:
            word = token
        correct_text = correct_text + ' ' + word
    tokens = tokenizer.tokenize(correct_text) 
    length = len(tokens)
    correct_text = ""
    i = 0
    while(i < length):

        if tokens[i] == '不':#中文乱码
            suggestions.update({i+1:tokens[i+1]})#给外部变量suggestions添加错误
            del tokens[i]
            length = length - 1
        elif tokens[i][0:2] == '##':
            word = tokens[i][2:]
            correct_text = correct_text + word  
            i = i+1
        else:
            token = tokens[i]
            if token not in ["'"]:
                word = ' '+ token
            else:
                word = token
                
            correct_text = correct_text + word  
            i = i+1
    return [correct_text]


def token_Align(tokens,text): 
    #tokens是拼写修正之后的文本的分词结果
    #text是原本可能带有拼写错误的文本
    #返回的是text的分词结果
    original_tokens = tokenizer.tokenize(text)
    original_tokens = ['[CLS]'] + original_tokens + ['[SEP]']
    print(original_tokens)
    length = len(tokens)
    i = 0
    while(i < min(length - 1,len(original_tokens) - 1)):
        tokens_length = min(length - 1,len(original_tokens) - 1)
        if original_tokens[i] == tokens[i] or (i+1<tokens_length and original_tokens[i+1] == tokens[i+1]) or (i+2<tokens_length and original_tokens[i+2] == tokens[i+2]) or (i+3<tokens_length and original_tokens[i+3] == tokens[i+3]):
            i = i+1
            continue
        else:
            if original_tokens[i][:2] == "##":
                original_tokens[i-1] = original_tokens[i-1] + original_tokens[i][2:]
                del original_tokens[i]
            elif original_tokens[i+1][:2] == "##":
                original_tokens[i] = original_tokens[i] + original_tokens[i+1][2:]
                del original_tokens[i+1]            
            elif tokens[i] == '[UNK]':
                original_tokens.insert(i,'[UNK]')
            else:
                if (i+1<tokens_length and original_tokens[i+1] == tokens[i]) or (i+2<tokens_length and original_tokens[i+2] == tokens[i+1]) or (i+3<tokens_length and original_tokens[i+3] == tokens[i+2]):
                    if re.match(r'[a-z]',original_tokens[i]) == None :
                        original_tokens[i] = original_tokens[i] + original_tokens[i+1]
                        del original_tokens[i+1] 
                    else:
                        pass #如果是单词交给下一个token处理，下一个可能是带##的token
                elif (i+1<tokens_length and original_tokens[i] == tokens[i+1]) or (i+2<tokens_length and original_tokens[i+1] == tokens[i+2]) or (i+3<tokens_length and original_tokens[i+2] == tokens[i+3]):
                    original_tokens.insert(i,' ')
                i = i + 1
                
    return original_tokens

def split_text(text0,threshold1,threshold2):#把文章分成一定长度的文段，保证GPU可以正常使用以及BERT模型不会超过最大的embeding
    texts = []
    text = ''
    tokens = text0[0].split(' ')
    count_tokens = 0
    last_HuanHang = -1
    new_tokens = []
    for token in tokens:
        if token == '':
            continue
        count_tokens = count_tokens + 1
        text = text + ' '+ token
        if (token == '.'and count_tokens > threshold1) or (token == '\r\n' and count_tokens > threshold2):
            texts.append([text])
            text = ''
            count_tokens = 0
    if count_tokens > 0:        
        texts.append([text])        
    return texts

# In[1619]:


import nltk
from pattern.en import conjugate, lemma, lexeme,PRESENT,SG
'''
    这是一个输出BERT模型训练结果的函数，方便查看调试
'''
def show_lm_probs(tokens, input_ids, probs, topk=5, firstk=20): #输出结果的函数，要最高概率topk个输出
    def print_pair(token, prob, end_str='', hit_mark=' '):
        if i < firstk:
            # token = token.replace('</w>', '').replace('\n', '/n')
            print('{}{: >3} | {: <12}'.format(hit_mark, int(round(prob*100)), token), end=end_str)
    
    ret = None
    for i in range(len(tokens)):
        ind_ = input_ids[i].item() if input_ids is not None else tokenizer.vocab[tokens[i]]
        prob_ = probs[i][ind_].item() #这个probs是该字符串第i个位置上填上词典上各个词的概率，prob_是词典上原来天的这个词的概率
        print_pair(tokens[i], prob_, end_str='\t')
        values, indices = probs[i].topk(topk)
        #print(values, indices)
        #print("****************************************************************************************************")
        top_pairs = []
        for j in range(topk):
            ind, prob = indices[j].item(), values[j].item()
            hit_mark = '*' if ind == ind_ else ' '
            token = tokenizer.ids_to_tokens[ind]
            print_pair(token, prob, hit_mark=hit_mark, end_str='' if j < topk - 1 else '\n')
            top_pairs.append((token, prob))
        if tokens[i] == "[MASK]":
            ret = top_pairs
    return ret 


# In[1621]:


def analyse_prob(prob,token):
    ind_ = tokenizer.vocab[token]
    prob_ = prob[ind_].item()
    top_prob = prob.max().item()
    top_ind = prob.argmax().item()
    top_word = tokenizer.ids_to_tokens[top_ind] #可能性最高的词
    gap = math.log(top_prob) - math.log(prob_) #计算两个词之间的差距 
    return top_word,gap


# In[1622]:


import colored
from colored import stylize
import spacy
nlp = spacy.load('en')
from nltk.corpus import wordnet as wn

def analyse_词性(token,tag):
    if 'VB' in tag: #如果是动词的各种时态
        tag0 = "v"
    elif "JJ" in tag : #形容词
        tag0 = "a"
    elif "RB" in tag: #副词
        tag0 = "r"
    elif "NN" in tag: #名词
        tag0 = "n"
    else:
        return tag
    if wn.morphy(token, tag0)==None:
        nlp = spacy.load('en')
        doc = nlp(token)
        tag = doc[0].tag_
    return tag
    
def show_abnormals(tokens,probs,text,show_suggestions=False): #多加了一个参数text，用来生成原来的token的
    global suggestions
    global original_tokens
    original_tokens = token_Align(tokens,text)
    def gap2color(mode):
        if mode == 1:
            return 'yellow_1'
        elif mode == 2:
            return 'orange_1'
        else:
            return 'red_1'
        
    def print_token(token, suggestion, gap ,mode):
        if gap == 0 and mode == 1:
            print(stylize(token + ' ', colored.fg('white') + colored.bg('black')), end='')
        else:
            print(stylize(token, colored.fg(gap2color(mode)) + colored.bg('black')), end='')
            if show_suggestions and mode > 1:
                print(stylize('/' + str(suggestion) + ' ', colored.fg('green' if gap > 10 else 'cyan') + colored.bg('black')), end='')
            else:
                print(stylize(' ', colored.fg(gap2color(mode)) + colored.bg('black')), end='')

        
    avg_gap = 0.
    tokens_tag = nltk.pos_tag(tokens) #给整个text做词性标注
    for i in range(1, len(tokens) - 1):  # skip first [CLS] and last [SEP]
        if tokens[i]=='[UNK]':
            continue
        top_word,gap = analyse_prob(probs[i],tokens[i])
        print()
        print("*******************************************************************************************************************")
        print(i)
        print(gap)
        avg_gap += gap
        suggestion = None
        #doc = nlp(tokens[i]) #用spacy标记
        #tag = doc[0].tag_ 
        #tag = nltk.pos_tag([tokens[i]])[0][1] #直接对token标记
        tag = tokens_tag[i][1]#当前tokens的词性，上面是用不同的方法标注词性
        tag = analyse_词性(tokens[i],tag)
        print(tag)
        
        if 'VB' in tag: #如果是动词的各种时态
            suggestion = analyse_V(i,probs[i],gap,top_word,2.5 ,7.9 ,1.8)
                
        elif "DT" == tag: #如果是冠词（冠词原则上不改变词性）
            suggestion = analyse_DT(i,probs[i],gap,top_word,3 ,4 ,1)
            
        elif "JJ" in tag : #形容词
            suggestion = analyse_adj(i,probs[i],gap,top_word,5 ,8 ,2)
                
        elif "RB" in tag: #副词
            suggestion = analyse_adv(i,probs[i],gap,top_word,5 ,8 ,2)
            
        elif "PRP" in tag: #代词
            suggestion = analyse_pronoun(i,probs[i],gap,top_word,3 ,5 ,1.5)
            
        elif "NN" in tag: #名词
            suggestion = analyse_N(i,probs[i],gap,top_word,4 ,10 ,2.2)
                    
        elif "CC" in tag: #连词
            suggestion = analyse_CC(i,probs[i],gap,top_word,2 ,2.5 ,1.5)
                
        elif "IN" == tag or 'TO' == tag: #介词
            suggestion = analyse_IN(i,probs[i],gap,top_word,3.5 ,4 ,1.5)
                
        elif 'MD' in tag: #情态动词
            suggestion = analyse_MD(i,probs[i],gap,top_word,3 ,4 ,1.5)
                
        elif "CD" in tag: #数词直接pass
            pass 
            
        elif "WDT" == tag and gap > 3.5: #who，which，that那些
            suggestion = top_word #推荐的词一般比较准
          
        elif tokens[i] in u',.!?[]()<>"\':，。！？【】（）《》“‘”’．':
            suggestion = analyse_biaodian(i,probs[i],gap,top_word,1.3 ,2 ,1)
            
        elif gap > 5:
            suggestion = top_word
        
        if (suggestion != None and suggestion.lower() != tokens[i] and suggestion.lower() != original_tokens[i]): #修改存在并且是另外一个词
            suggestions.update({i:suggestion})
            mode = 2
        elif suggestions.__contains__(i)==True: #这是因为之前在拼写检查时已经修改了该位置的单词
            if original_tokens[i] == tokens[i]:
                del suggestions[i]
                mode = 1
            else:
                mode = 2
                suggestion = suggestions[i]
        else:
            if original_tokens[i] != tokens[i]:
                mode = 2
                suggestions[i] = tokens[i]
                suggestion = tokens[i]
            else:
                mode = 1
        
        print_token(original_tokens[i], suggestion, gap, mode)
        print()
        print(original_tokens[i],tokens[i],suggestion,mode)
    avg_gap /= (len(tokens) - 2)
    print()
    print('平均gap:'+ str(avg_gap))
    return avg_gap

def analyze_part_text(text, masked_tokens=None, show_suggestions=True, show_firstk_probs=500):
    print("原始文本")
    print(text)
    step = 15 #用于训练加速的步长，每15个token被mask一个位置
    global input_ids_sen,input_type_ids_sen,in_sentence,sentences,entire_ids,entire_type_ids,suggestions,original_tokens
    suggestions = {}#清空全局变量
    text = process_biaodian(text)
    print("标点处理后")
    print(text)
    text0 = text  #保存有拼写错误的文本
    text = correct_spelling(text[0]) #拼写修正过得文本
    print("拼写修正后********************************")
    print(text)
    print("********************************")
    #黄金搭档token_Align放在show_abnormals里面了
    input_ids_sen,input_type_ids_sen,in_sentence,sentences,entire_ids,entire_type_ids = process_text(text[0])
    
    examples = convert_text_to_examples(text)
    features = convert_examples_to_features(examples, tokenizer, print_info=False)
    given_mask = "[MASK]" in features[0].tokens
    if not given_mask or masked_tokens is not None:
        assert len(features) == 1
        features, batches = copy_and_mask_feature(features[0],step, masked_tokens=masked_tokens)
        #print(len(features))

    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long) #把input_ids增加了一个维度，变成[n_features,sequence_len]
    #这里的n_features实际上是句子有多少批训练

    input_type_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long) #把input_type_ids增加了一个维度，其实每一行都一样
    input_ids = input_ids.to(device) 
    input_type_ids = input_type_ids.to(device)

    mlm_logits = model(input_ids)
    mlm_probs = F.softmax(mlm_logits, dim=-1) 
    tokens = features[0].tokens #为了输出，[mask]在input_ids里面表示出来，features的token都一样
    print(tokens)
    if not given_mask or masked_tokens is not None:
        bsz, seq_len, vocab_size = mlm_probs.size() #三个维度分别是batch_size, sequence_length, vocab_size
        assert bsz == len(batches)
        reduced_mlm_probs = torch.Tensor(1, len(tokens), vocab_size)
        for i in batches:
            pos = i
            while pos < len(tokens):
                reduced_mlm_probs[0, pos] = mlm_probs[i, pos]
                pos = pos + step
        mlm_probs = reduced_mlm_probs #压缩一下大小，节约不必要浪费的空间（只需要第i个batch里面[mask]位置的词汇表概率即可）
    top_pairs = show_lm_probs(tokens, None, mlm_probs[0], firstk=show_firstk_probs) #传入的probs是二维的
    if not given_mask:
        avg_gap = show_abnormals(tokens,mlm_probs[0],text0[0], show_suggestions=show_suggestions)
    return suggestions,original_tokens,avg_gap


def analyze_text(text, masked_tokens=None, show_suggestions=True, show_firstk_probs=500):
    suggestions = {}
    avg_gap = 0
    new_part_suggestions = {}
    original_tokens = ['[CLS]','[SEP]']
    text = process_biaodian(text)
    text0 = text  #保存有拼写错误的文本
    texts = split_text(text,130,100)
    accumulate_length = 0
    remainer = 2 #[CLS]和[SEP]
    for text0 in texts:
        part_suggestions,part_original_tokens,part_avg_gap = analyze_part_text(text0, masked_tokens, show_suggestions, show_firstk_probs)
        for key in part_suggestions:
            new_part_suggestions[key + accumulate_length] = part_suggestions[key]
        tokens_length = len(part_original_tokens)
        accumulate_length = accumulate_length + tokens_length - remainer
        suggestions.update(new_part_suggestions)
        original_tokens = original_tokens[:-1] + part_original_tokens[1:]
        avg_gap = avg_gap + part_avg_gap*(tokens_length - 2)
    avg_gap = avg_gap/(accumulate_length)
    return suggestions,original_tokens,avg_gap
# In[1626]:



'''
    功能：对suggestions进行修改，由于某处位置改变造成suggestions后面的错误位置都相应移动
    输入：
        index：开始移动的位置
        direction：移动的方向，1表示向右边移，-1表示向左边移
'''
def modify_suggestions(index,direction):
    global suggestions
    new_suggestions = {};
    if direction == 0:
        pass
    elif direction == 1:
        for key in suggestions:
            if key < index:
                new_suggestions.update({key:suggestions[key]})
            else:
                new_suggestions.update({key+1:suggestions[key]})
    elif direction == -1:
        for key in suggestions:
            if key < index:
                new_suggestions.update({key:suggestions[key]})
            else:
                new_suggestions.update({key-1:suggestions[key]})       
    suggestions = new_suggestions


# In[1592]:


#print(suggestions)
def display_suggestion():
    print("**********************************display_suggestions********************************************************")
    print("| {:50} : {}".format("suggestion","position in text"))
    print("---------------------------------------------------------------------------------------")
    for key in suggestions:
        print("| {:<50} : {}".format(suggestions[key] ,key))
    print("*************************************************************************************************************")
#display_suggestion()

'''
    功能：
        修改文本，tokens，suggestions
    输入：
        index：修改的位置
        text：被修改前的原文
    输出：
        [text]：修改后的文本
        new_tokens：修改后的新tokens
        suggestions：修改后新的建议字典
'''
def modify_text(index,text): #修改文本，tokens，以及suggestions
    global suggestions,original_tokens
    tokens = original_tokens
    new_text = ""
    suggestion = suggestions[index]
    del(suggestions[index])
    suggestion_tokens = suggestion.split(" ")
    #print(suggestion_tokens)
    if '去掉前面' == suggestion_tokens[0]:
        del tokens[index - 1]
        del suggestion_tokens[0]
        del suggestion_tokens[0]
        modify_suggestions(index,-1)
        index = index - 1
    elif '去掉后面' == suggestion_tokens[0]:
        del tokens[index + 1]
        del suggestion_tokens[0]
        del suggestion_tokens[0]
        modify_suggestions(index+2,-1)
    elif '去掉' == suggestion_tokens[0]:
        del tokens[index]
        del suggestion_tokens[0]
        del suggestion_tokens[0]
        modify_suggestions(index+1,-1)
    if '原位置改成' in suggestion_tokens:
        del suggestion_tokens[0]
        
        
    len_suggest = len(suggestion_tokens)
    if len_suggest == 1:
        tokens[index] = suggestion_tokens[0]
    elif len_suggest == 2:
        tokens.insert(index,suggestion_tokens[0])
        tokens[index + 1] = suggestion_tokens[1]
        modify_suggestions(index+1,1)
    final_len = len(tokens)

    for i in range(1,len(tokens)-1):
        word = tokens[i]
        if word[0:2] == "##":
            new_text = new_text + word[2:]
        else:
            new_text = new_text + ' ' + word
            
    original_tokens = tokens
    return [text],tokens,suggestions


# In[1576]:


#变成py文件
try:
    get_ipython().system('jupyter nbconvert --to python likunlin_final.ipynb')
except:
    pass

