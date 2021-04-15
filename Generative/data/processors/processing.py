# -*- coding: utf-8 -*-
# -*- author: JeremySun -*-
# -*- dating: 21/3/2 -*-

"""This is a preprocessor for ConvQA_CN_v3.0 dataset."""

import os
import sys
import json
import string
import collections
import regex as re
from collections import Counter
import logging
import tqdm

sys.path.append(r'E:\Internship\ConvQA\Reference\GenerativeCoQA\s2s_ft/')
from transformers.tokenization_bert import whitespace_tokenize, BertTokenizer

tokenizer = BertTokenizer.from_pretrained(r"C:\pretrained_model\unilm-cn-base/", do_lower_case=True)

coqa_train = "ConvQA_data/ConvQA_CN_v3.0_train.json"  # CoQA原训练集
coqa_dev = "ConvQA_data/ConvQA_CN_v3.0_dev.json"  # CoQA原验证集

CoQA_train_json = "./QAQAFQ/ConvQA_train.json"  # mode='w'
CoQA_dev_json = "./QAQAFQ/ConvQA_dev.json"  # mode='w'

logger = logging.getLogger(__name__)

max_src_length = 509  # 467 + 3 = 470 | 509 + 3 = 512
doc_stride = 128
max_question_len = 60


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def len_preserved_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def len_preserved_space(matchobj):
        return ' ' * len(matchobj.group(0))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', len_preserved_space, text)

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return remove_articles(remove_punc(lower(s)))


def split_with_span(s):
    if s.split() == []:
        return [], []
    else:
        return zip(*[(m.group(0), (m.start(), m.end() - 1)) for m in re.finditer(r'\S+', s)])


def free_text_to_span(free_text, full_text):
    if free_text == "无法作答":
        return "__NA__", -1, -1
    if normalize_answer(free_text) == "是的":
        return "__YES__", -1, -1
    if normalize_answer(free_text) == "不是":
        return "__NO__", -1, -1

    free_ls = len_preserved_normalize_answer(free_text).split()
    full_ls, full_span = split_with_span(len_preserved_normalize_answer(full_text))
    if full_ls == []:
        return full_text, 0, len(full_text)

    max_f1, best_index = 0.0, (0, len(full_ls) - 1)
    free_cnt = Counter(free_ls)
    for i in range(len(full_ls)):
        full_cnt = Counter()
        for j in range(len(full_ls)):
            if i + j >= len(full_ls):
                break
            full_cnt[full_ls[i + j]] += 1

            common = free_cnt & full_cnt
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / (j + 1)
            recall = 1.0 * num_same / len(free_ls)
            f1 = (2 * precision * recall) / (precision + recall)

            if max_f1 < f1:
                max_f1 = f1
                best_index = (i, j)

    assert (best_index is not None)
    (best_i, best_j) = best_index
    char_i, char_j = full_span[best_i][0], full_span[best_i + best_j][1] + 1

    return full_text[char_i:char_j], char_i, char_j


def find_lcsubstr(s1, s2):
    """返回最长公共子串以及最长匹配长度"""
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]  # 生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax = 0  # 最长匹配的长度
    p = 0  # 最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:  # 如果相等，则加入现有的公共子串
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax  # 返回最长子串及其长度


def proprecess(file, writer, is_training=True):
    with open(file, "r", encoding='utf-8') as reader:
        source = json.load(reader)
        input_data = source["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F or c == '\xa0':
            return True
        return False

    examples = []
    for paragraph in tqdm.tqdm(input_data):  # 这里的paragraph相当于datum
        paragraph_text = paragraph["Passage"].lower()
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True  # 之前的为空白符
        for c in paragraph_text:  # 按字符读取
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:  # 如果前面为空格, 那么新增一个单词
                    doc_tokens.append(c)
                else:  # 否则仍为上一单词, 那么string拼接
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)  # 统计story中  [0,0,0,1,1,1,2,2,... ]

        for qa_idx, q in enumerate(paragraph["Questions"]):
            """bug: 所有generative都没有用处理进train.json和dev.json"""
            qas_id = str(paragraph['id']) + '#' + str(q["turn_id"])  # qasid = xxx#1
            if qa_idx > 1:  # 记录前一对话q、a和当前对话的q
                question_text = '{} {} {} {} {} {}'.format(
                    paragraph['Questions'][qa_idx - 2]['input_text'],
                    paragraph['Answers'][qa_idx - 2]['input_text'],  # 这里使用的是'input_text'
                    paragraph['Questions'][qa_idx - 1]['input_text'],
                    paragraph['Answers'][qa_idx - 1]['input_text'],  # 这里使用的是'input_text'
                    paragraph['Answers'][qa_idx - 1]['input_text'],  # 这里添加了'span_text'
                    q['input_text'])
                query = paragraph['Answers'][qa_idx - 1]['input_text'] + q['input_text']
            elif qa_idx > 0:
                question_text = '{} {} {} {}'.format(paragraph['Questions'][qa_idx - 1]['input_text'],
                                                     paragraph['Answers'][qa_idx - 1]['input_text'],
                                                     paragraph['Answers'][qa_idx - 1]['input_text'],
                                                     q['input_text'])
                query = paragraph['Answers'][qa_idx - 1]['input_text'] + q['input_text']
            else:  # 记录第一个q.
                question_text = q["input_text"]
                query = q["input_text"]

            start_position = None
            end_position = None
            orig_answer_text = None

            answer_text = paragraph['Answers'][qa_idx]['input_text'].lower()
            span_text = paragraph['Answers'][qa_idx]['input_text'].lower()
            orig_answer_text, char_i, char_j = free_text_to_span(answer_text, span_text)  # 取answer在span中的最佳位置.

            ans_choice = 0 if orig_answer_text == '__NA__' else \
                1 if orig_answer_text == '__YES__' else \
                    2 if orig_answer_text == '__NO__' else \
                        3  # Not a yes/no question

            """对于生成式答案的开始位置和结束位置需谨慎，不能成为很大的噪声"""

            # 这样的预处理方式跳过了所有的generative
            # if ans_choice == 3:
            #     if paragraph['Answers'][qa_idx]['answer_type'] == 'extractive':
            #         init_answer_offset = paragraph['Answers'][qa_idx]['answer_start'][0]
            #     else:
            #         init_answer_offset = paragraph['Passage'].find(paragraph['Answers'][qa_idx]['input_text'])
            #     answer_offset = init_answer_offset + char_i  # 默认answer_start列表里的第一个数字为答案的开始位置
            #     answer_length = len(orig_answer_text)
            #     start_position = char_to_word_offset[answer_offset]  # 记录最佳answer 在story中起始位置  这边记录的就是token的位置,不具体到字符
            #     end_position = char_to_word_offset[answer_offset + answer_length - 1]  # 记录最佳answer 在story中终止位置
            #
            #     actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])  # 最匹配的token范围  word级别 'i LOVE YOU'
            #     cleaned_answer_text = " ".join(
            #         whitespace_tokenize(orig_answer_text))  # orig_answer_text  character级别 可能为'i LOVE Y'
            #     if actual_text.find(cleaned_answer_text) == -1:
            #         print(whitespace_tokenize(orig_answer_text), doc_tokens[start_position: (end_position + 1)])
            #         logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
            #         continue
            # else:
            #     start_position = -1
            #     end_position = -1
            #     orig_answer_text = ""

            # 用find()和最长公共子串去修正
            if ans_choice == 3:
                if paragraph['Answers'][qa_idx]['answer_type'] == 'extractive':
                    init_answer_offset = paragraph['Answers'][qa_idx]['answer_start'][0]
                    answer_offset = init_answer_offset + char_i  # 默认answer_start列表里的第一个数字为答案的开始位置
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]  # 记录最佳answer 在story中起始位置  这边记录的就是token的位置,不具体到字符
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]  # 记录最佳answer 在story中终止位置
                else:  # find()有可能是返回-1，那么orig_to_tok_index会出现IndexError
                    generative_index = paragraph['Passage'].find(paragraph['Answers'][qa_idx]['input_text'])
                    if generative_index != -1:
                        init_answer_offset = generative_index
                        answer_offset = init_answer_offset + char_i
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    else:
                        # 不只用find()，配合find_lcsubstr()返回最长公共子串以及子串长度
                        find_substring_idx = find_lcsubstr(s1=paragraph['Passage'], s2=paragraph['Answers'][qa_idx]['input_text'])
                        if find_substring_idx:
                            longest_substring = find_substring_idx[0]
                            longest_substring_len = find_substring_idx[1]
                            init_answer_offset = paragraph['Passage'].find(longest_substring)
                            answer_offset = init_answer_offset + char_i
                            # answer_length = len(answer_offset)
                            start_position = char_to_word_offset[answer_offset]
                            end_position = char_to_word_offset[answer_offset + longest_substring_len - 1]
                        else:  # 这里说明这类答案完全由用户原创，这里为了后续计算方便，默认开始位置和结束位置均为0或者-1，尽量避免其对抽取式答案类型的影响
                            start_position = 0
                            end_position = 0
            else:
                start_position = -1
                end_position = -1
                orig_answer_text = ""
            query_tokens = tokenizer.tokenize(question_text.lower())
            if len(query_tokens) > max_question_len:
                query_tokens = query_tokens[-max_question_len:]
            answer_query = tokenizer.tokenize(query.lower())

            tok_to_orig_index = []
            orig_to_tok_index = []
            all_doc_tokens = []
            for (i, token) in enumerate(doc_tokens):
                orig_to_tok_index.append(len(all_doc_tokens))
                sub_tokens = tokenizer.tokenize(token)
                for sub_token in sub_tokens:
                    tok_to_orig_index.append(i)
                    all_doc_tokens.append(sub_token)
            tok_start_position = None
            tok_end_position = None

            if ans_choice != 3:
                tok_start_position = -1
                tok_end_position = -1
            else:
                tok_start_position = orig_to_tok_index[start_position]
                if end_position < len(doc_tokens) - 1:
                    tok_end_position = orig_to_tok_index[end_position + 1] - 1
                else:
                    tok_end_position = len(doc_tokens) - 1

            max_tokens_for_doc = max_src_length - len(query_tokens) - 2
            # print(max_tokens_for_doc)

            _DocSpan = collections.namedtuple("DocSpan", ["start", "length"])
            doc_spans = []
            start_offset = 0
            while start_offset < len(all_doc_tokens):
                length = len(all_doc_tokens) - start_offset
                if length > max_tokens_for_doc:  # doc超出max就截断
                    length = max_tokens_for_doc
                doc_spans.append(_DocSpan(start=start_offset, length=length))  # 记录分段的起始和长度
                if start_offset + length == len(all_doc_tokens):
                    break
                start_offset += min(length, doc_stride)
            # print(doc_spans)

            best_span_idx = 0
            best_count = 0

            for (doc_span_index, doc_span) in enumerate(doc_spans):
                count = 0
                span = " ".join(all_doc_tokens[doc_span.start:doc_span.start + doc_span.length - 1])
                for i in answer_query:
                    count += span.count(i, 0, len(span) - 1)
                if count > best_count:
                    best_count = count
                    best_span_idx = doc_span_index

            if ans_choice == 3:
                doc_start = doc_spans[best_span_idx].start
                doc_end = doc_spans[best_span_idx].start + doc_spans[best_span_idx].length - 1

                out_of_span = False
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    # 重定位
                    start_position = -1
                    end_position = -1

                    if is_training:
                        for (doc_span_index, doc_span) in enumerate(doc_spans):
                            doc_start = doc_spans[doc_span_index].start
                            doc_end = doc_spans[doc_span_index].start + doc_spans[doc_span_index].length - 1
                            if tok_start_position >= doc_start and tok_end_position <= doc_end:
                                best_span_idx = doc_span_index
                                doc_offset = len(query_tokens) + 1
                                start_position = tok_start_position - doc_start + doc_offset
                                end_position = tok_end_position - doc_start + doc_offset
                else:
                    doc_offset = len(query_tokens) + 1
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset
            else:
                start_position = -1
                end_position = -1

            ans_tokens = tokenizer.tokenize(paragraph['Answers'][qa_idx]['input_text'].lower())

            dic = {}
            src = []
            for i in query_tokens:
                src.append(i)
            src.append("[SEP]")
            for x in all_doc_tokens[doc_spans[best_span_idx].start: doc_spans[best_span_idx].start + doc_spans[best_span_idx].length - 1]:
                src.append(x)
            dic["src"] = src
            if len(ans_tokens) > 42:
                ans_tokens = ans_tokens[:42]
            dic["tgt"] = ans_tokens
            dic["start_position"] = start_position + 1
            dic["end_position"] = end_position + 1
            dic["ans_choice"] = ans_choice
            dic["id"] = qas_id
            if not is_training:
                dic["passages"] = paragraph_text
            json.dump(dic, writer, ensure_ascii=False)
            writer.write("\n")


if __name__ == "__main__":
    # train data
    train = open(CoQA_train_json, "w", encoding="utf-8")
    proprecess(coqa_train, train)

    # dev data
    dev = open(CoQA_dev_json, "w", encoding="utf-8")
    proprecess(coqa_dev, dev, False)

    # """this is a test."""
    # test_file = "./QAQAFQ/ConvQA_train_test.json"  # mode='w'
    # train = open(test_file, "w", encoding="utf-8")
    # proprecess(coqa_train, train, is_training=True)
