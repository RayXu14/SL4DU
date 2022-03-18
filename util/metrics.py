from collections import Counter
import re

import numpy        
import nltk
from nltk import ngrams
from nltk.translate import bleu_score as nltkbleu
from stop_words import get_stop_words



def recall_2at1(score_list, k=1):
    num_correct = 0
    num_total = len(score_list)
    for scores in score_list:
        ranking_index = numpy.argsort(-numpy.array(scores[0:2]))
        # Message at index 0 is always correct in our test data
        if 0 in ranking_index[:k]:
            num_correct += 1
    return float(num_correct) / num_total


def recall_at_k(labels, scores, k=1, doc_num=10):
    scores = scores.reshape(-1, doc_num) # [batch, doc_num]
    labels = labels.reshape(-1, doc_num) # [batch, doc_num]
    sorted = numpy.sort(scores, 1)
    indices = numpy.argsort(-scores, 1)
    count_nonzero = 0
    recall = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel == 0:
            continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        recall += float(rel) / float(num_rel)
        count_nonzero += 1
    return float(recall) / count_nonzero


def precision_at_k(labels, scores, k=1, doc_num=10):
    
    scores = scores.reshape(-1, doc_num) # [batch, doc_num]
    labels = labels.reshape(-1, doc_num) # [batch, doc_num]
    sorted = numpy.sort(scores, 1)
    indices = numpy.argsort(-scores, 1)
    count_nonzero = 0
    precision = 0
    for i in range(indices.shape[0]):
        num_rel = numpy.sum(labels[i])
        if num_rel == 0:
            continue
        rel = 0
        for j in range(k):
            if labels[i, indices[i, j]] == 1:
                rel += 1
        precision += float(rel) / float(k)
        count_nonzero += 1
    return precision / count_nonzero


def MAP(target, logits, k=10):
    """
    Compute mean average precision.
    :param target: 2d array [batch_size x num_clicks_per_query] true
    :param logits: 2d array [batch_size x num_clicks_per_query] pred
    :return: mean average precision [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)
    sorted = numpy.sort(logits, 1)[::-1]
    indices = numpy.argsort(-logits, 1)
    count_nonzero = 0
    map_sum = 0
    for i in range(indices.shape[0]):
        average_precision = 0
        num_rel = 0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                num_rel += 1
                average_precision += float(num_rel) / (j + 1)
        if num_rel==0:
            continue
        average_precision = average_precision / num_rel
        map_sum += average_precision
        count_nonzero += 1
    return float(map_sum) / count_nonzero


def MRR(target, logits, k=10):
    """
    Compute mean reciprocal rank.
    :param target: 2d array [batch_size x rel_docs_per_query]
    :param logits: 2d array [batch_size x rel_docs_per_query]
    :return: mean reciprocal rank [a float value]
    """
    assert logits.shape == target.shape
    target = target.reshape(-1,k)
    logits = logits.reshape(-1,k)
    sorted = numpy.sort(logits, 1)[::-1]
    indices = numpy.argsort(-logits, 1)
    count_nonzero = 0
    reciprocal_rank = 0
    for i in range(indices.shape[0]):
        flag=0
        for j in range(indices.shape[1]):
            if target[i, indices[i, j]] == 1:
                reciprocal_rank += float(1.0) / (j + 1)
                flag=1
                break
        if flag:
            count_nonzero += 1
    return float(reciprocal_rank) / count_nonzero


def auto_report_RS(all_labels, all_preds, dataset):
    list_r_1_2 = []
    list_r_1_10 = []
    list_r_2_10 = []
    list_r_5_10 = []
    list_p_1 = []
    list_map = []
    list_mrr = []

    all_labels = numpy.array(all_labels, dtype=int)
    all_preds = numpy.array(all_preds, dtype=float)
    for i in range(0, len(all_labels), 10):
        ts = all_labels[i:i + 10]
        ps = all_preds[i:i + 10]
        if sum(ts) == 0:
            continue
        score_list = numpy.split(ps, 1, axis=0)
        recall_2_1  = recall_2at1(score_list)
        recall_at_1 = recall_at_k(ts, ps, 1) 
        recall_at_2 = recall_at_k(ts, ps, 2)
        recall_at_5 = recall_at_k(ts, ps, 5)
        p_at_1 = precision_at_k(ts, ps)
        map_ = MAP(ts, ps)
        mrr_ = MRR(ts, ps)
        list_r_1_2.append(recall_2_1) 
        list_r_1_10.append(recall_at_1)
        list_r_2_10.append(recall_at_2) 
        list_r_5_10.append(recall_at_5) 
        list_p_1.append(p_at_1) 
        list_map.append(map_) 
        list_mrr.append(mrr_)

    report = '\n'.join([f'MAP    = {numpy.average(list_map)}',
                        f'MRR    = {numpy.average(list_mrr)}',
                        f'P_1    = {numpy.average(list_p_1)}',
                        f'R_1_2  = {numpy.average(list_r_1_2)}',
                        f'R_1_10 = {numpy.average(list_r_1_10)}',
                        f'R_2_10 = {numpy.average(list_r_2_10)}',
                        f'R_5_10 = {numpy.average(list_r_5_10)}'])
                        
    if dataset == 'Douban':
        main_metric = numpy.average(list_p_1)
    else:
        main_metric = numpy.average(list_r_1_10)
    return report, main_metric
        

def bleu_corpus(hypothesis, references):
    from nltk.translate.bleu_score import corpus_bleu
    hypothesis = hypothesis.copy()
    references = references.copy()
    hypothesis = [hyp.split() for hyp in hypothesis]
    references = [[ref.split()] for ref in references]
    # hypothesis = [normalize_answer(hyp).split(" ") for hyp in hypothesis]
    # references = [[normalize_answer(ref).split(" ")] for ref in references]
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=nltkbleu.SmoothingFunction(epsilon=1e-12).method1)
    return (b1, b2, b3, b4)

    
def bleu_metric(hypothesis, references):
    return bleu_corpus(hypothesis, references)


def distinct_metric(hypothesis):
    '''
    compute distinct metric
    :param hypothesis: list of str
    :return:
    '''
    unigram_counter, bigram_counter = Counter(), Counter()
    for hypo in hypothesis:
        tokens = hypo.split()
        unigram_counter.update(tokens)
        bigram_counter.update(ngrams(tokens, 2))

    distinct_1 = len(unigram_counter) / (sum(unigram_counter.values()) + 1e-20)
    distinct_2 = len(bigram_counter) / (sum(bigram_counter.values()) + 1e-20)
    return distinct_1, distinct_2

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))