
import numpy as np
import scipy.sparse
import re
import scipy.io
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx


from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as sklearn_stopwords
from nltk import word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet



def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None
def load_glove_vectors(dim=50):
    print('Loading GloVe pretrained word vectors')
    file_paths = {
        50: 'utils/wordvec/GloVe/glove.6B.50d.txt',
        100: 'wordvec/GloVe/glove.6B.100d.txt',
        200: 'wordvec/GloVe/glove.6B.200d.txt',
        300: 'wordvec/GloVe/glove.6B.300d.txt'
    }
    f = open(file_paths[dim], 'r', encoding='utf-8')
    wordvecs = {}
    for line in f.readlines():
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        wordvecs[word] = embedding
    print('Done.', len(wordvecs), 'words loaded!')
    return wordvecs

# glove_dim = 50
# glove_vectors = load_glove_vectors(dim=glove_dim)

author_select = pd.read_csv('DBLP/DBLP_author_2008.txt', sep='\t', header=None, names=['author_id', 'author_name'], keep_default_na=False, encoding='utf-8')
paper_author = pd.read_csv('DBLP/DBLP_paper_author.txt', sep='\t', header=None, names=['paper_id', 'author_id'], keep_default_na=False, encoding='utf-8')
paper_conf = pd.read_csv('DBLP/DBLP_paper_conf.txt', sep='\t', header=None, names=['paper_id', 'conf_id'], keep_default_na=False, encoding='utf-8')
paper_term = pd.read_csv('DBLP/DBLP_paper_term.txt', sep='\t', header=None, names=['paper_id', 'term_id'], keep_default_na=False, encoding='utf-8')
paper_year = pd.read_csv('DBLP/DBLP_paper_year.txt', sep='\t', header=None, names=['paper_id', 'year'], keep_default_na=False, encoding='utf-8')
papers = pd.read_csv('DBLP/DBLP_paper.txt', sep='\t', header=None, names=['paper_id', 'paper_title','paper_abstract'], keep_default_na=False, encoding='utf-8')
terms = pd.read_csv('DBLP/DBLP_term.txt', sep='\t', header=None, names=['term_id', 'term'], keep_default_na=False, encoding='utf-8')
confs = pd.read_csv('DBLP/DBLP_conf.txt', sep='\t', header=None, names=['conf_id', 'conf'], keep_default_na=False, encoding='utf-8')

se_authors = author_select['author_id'].to_list()  # 2008前有历史的
paper_author = paper_author[paper_author['author_id'].isin(se_authors)].reset_index(drop=True)
valid_papers = paper_author['paper_id'].unique()
print(len(se_authors))
print(len(valid_papers))
print(len(papers['paper_id'].isin(valid_papers)))
papers = papers[papers['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_conf = paper_conf[paper_conf['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_term = paper_term[paper_term['paper_id'].isin(valid_papers)].reset_index(drop=True)
paper_year = paper_year[paper_year['paper_id'].isin(valid_papers)].reset_index(drop=True)
valid_terms = paper_term['term_id'].unique()
terms = terms[terms['term_id'].isin(valid_terms)].reset_index(drop=True)
print(len(terms))

import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')


lemmatizer = WordNetLemmatizer()
lemma_id_mapping = {}
lemma_list = []
lemma_id_list = []
i = 0
for _, row in terms.iterrows():
    i += 1
    lemma = lemmatizer.lemmatize(row['term'])
    lemma_list.append(lemma)
    if lemma not in lemma_id_mapping:
        lemma_id_mapping[lemma] = row['term_id']
    lemma_id_list.append(lemma_id_mapping[lemma])
terms['lemma'] = lemma_list
terms['lemma_id'] = lemma_id_list

term_lemma_mapping = {row['term_id']: row['lemma_id'] for _, row in terms.iterrows()}
lemma_id_list = []

for _, row in paper_term.iterrows():
    lemma_id_list.append(term_lemma_mapping[row['term_id']])
paper_term['lemma_id'] = lemma_id_list

paper_term = paper_term[['paper_id', 'lemma_id']]
paper_term.columns = ['paper_id', 'term_id']
paper_term = paper_term.drop_duplicates()
terms = terms[['lemma_id', 'lemma']]
terms.columns = ['term_id', 'term']
terms = terms.drop_duplicates()

#

author_select = author_select.sort_values('author_id').reset_index(drop=True)  # 8505
papers = papers.sort_values('paper_id').reset_index(drop=True)
terms = terms.sort_values('term_id').reset_index(drop=True)
confs = confs.sort_values('conf_id').reset_index(drop=True)
dim = len(author_select) + len(papers) + len(terms) + len(confs)

author_id_mapping = {row['author_id']: i for i, row in author_select.iterrows()}
paper_id_mapping = {row['paper_id']: i + len(author_select) for i, row in papers.iterrows()}
term_id_mapping = {row['term_id']: i + len(author_select) + len(papers) for i, row in terms.iterrows()}
conf_id_mapping = {row['conf_id']: i + len(author_select) + len(papers) + len(terms) for i, row in confs.iterrows()}

# DBLP要用历史paper yelp不需要
his_papers = paper_year[paper_year['year'] < 2008].reset_index(drop=True)
his_id = his_papers['paper_id'].unique()
train_papers = papers[papers['paper_id'].isin(his_id)].reset_index(drop=True)   

paper_content = train_papers['paper_title'].values+train_papers['paper_abstract'].values

print(paper_content.shape)

p_time = {row['paper_id']: int(row['year']) for i, row in paper_year.iterrows()}
print(len(p_time))
print(len(paper_id_mapping))
import copy

np.random.seed(123)
split_year =2008
adjM = np.zeros((dim, dim), dtype=int)
for _, row in paper_author.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = author_id_mapping[row['author_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_term.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = term_id_mapping[row['term_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1
for _, row in paper_conf.iterrows():
    idx1 = paper_id_mapping[row['paper_id']]
    idx2 = conf_id_mapping[row['conf_id']]
    adjM[idx1, idx2] = 1
    adjM[idx2, idx1] = 1


paper_test = [paper_id_mapping[p] for p in p_time if p_time[p] >= split_year]
p_id_end = len(author_select) + len(papers)

for p in paper_test:  # 
    adjM[p] = [0] * len(adjM)
    adjM[:, p] = [0] * len(adjM)
count = 0
for s in adjM[11569:11248+11569]:
    if sum(s) > 0:
        count += 1
print(count)

stopwords = sklearn_stopwords.union(set(nltk_stopwords.words('english')))
paper_words = []
for p in paper_content:
    _ = re.sub(r'[^\w\s]', ' ', p.lower()).split()
    words = [word for word in _ if word not in stopwords and not word.isdigit()]
    words = pos_tag(words)
    lemmas_sent = []
    for word in words:
        wordnet_pos = get_wordnet_pos(word[1]) or wordnet.NOUN
        lemmas_sent.append(lemmatizer.lemmatize(word[0], pos=wordnet_pos))
    paper_words.append(lemmas_sent)

for num_topics in [32, 64]:
    print(num_topics)
    dictionary = Dictionary(paper_words)

    corpus = [dictionary.doc2bow(text) for text in paper_words]

    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
    topic_list = lda.print_topics(num_topics)
    print("10个主题的单词分布为：\n")
    for topic in topic_list:
        print(topic)

    topic_lamda = lda.get_document_topics(corpus)
    perplexity = lda.log_perplexity(corpus)
    print(perplexity)


    print(len(topic_lamda))  # 7544  历史大小
    lamda = np.zeros([len(paper_id_mapping), num_topics])    
    for i, p_topic in enumerate(topic_lamda):
        for t in p_topic:  # 具体概率
            lamda[paper_id_mapping[train_papers['paper_id'].values[i]]-len(author_select), t[0]] = t[1]
    save_prefix = 'DBLP/DBLP_preprocess/'+str(num_topics)
    lamda = scipy.sparse.csr_matrix(lamda)
    scipy.sparse.save_npz(save_prefix + '_text_topic.npz', lamda)



