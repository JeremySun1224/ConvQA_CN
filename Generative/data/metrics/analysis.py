
"""缺少一个s2s_mlm_rule.json"""

import argparse
import json
import re
import string
import sys
from collections import Counter, OrderedDict

dev_json = "V0/dev.json"  # mode='r'
gold_file = "CoQA_data/coqa_dev_all_answer.txt"  # mode='r'
pred_file = "CoQA_output/s2s_mlm_rule.json"  # mode='r'
analysis_file = "CoQA_s2s.json"  # mode='w'


def normalize_answer(s):
    """Lower text and remove punctuation, storys and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1_all(a_gold, a_pred):
    max_f1 = 0.0
    pred_toks = get_tokens(a_pred)
    for x in a_gold.split("[SEP]"):
        f1 = 0
        x = x.strip().replace("\n", "")
        gold_toks = get_tokens(x)
        common = Counter(gold_toks) & Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            # print("error!")
            f1 = int(gold_toks == pred_toks)
        elif num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2.0 * precision * recall) / (precision + recall)
        if f1 > max_f1:
            max_f1 = f1
        else:
            continue

    return max_f1


def main():
    gold_list = []
    with open(gold_file, "r", encoding="utf-8") as reader:
        for x in reader:
            gold_list.append(x)
    reader.close()

    pred_list = []
    with open(pred_file, "r", encoding="utf-8") as reader:
        pred_list = json.load(reader)
    reader.close()

    dev_list = []
    with open(dev_json, "r", encoding="utf-8") as reader:
        for line in reader:
            dev_list.append(json.loads(line))
    reader.close()

    F1 = 0.0
    numynu = 0  # number of yes/no/unknown
    kk = 0
    t = {"paragraphs": []}
    for i in range(len(gold_list)):
        f = compute_f1_all(gold_list[i], pred_list[i]["answer"])
        F1 += f
        if f < 1:
            pred_toks = get_tokens(pred_list[i]["answer"])
            three = ["yes", 'unknown', "no"]
            for x in three:
                if x in pred_toks:
                    numynu += 1
            pred = pred_list[i]["answer"].replace(" ", "")
            pred = str(pred.strip())
            if pred == gold_list[i].split("[SEP]")[0]:
                kk += 1
            s = {}
            s["context"] = " ".join(dev_list[i]["src"]).split("[SEP]")[1]
            s["question"] = " ".join(dev_list[i]["src"]).split("[SEP]")[0]
            gold_answer = []
            for x in gold_list[i].split("[SEP]"):
                x = x.replace("\n", "")
                gold_answer.append(x)
            s["gold_answer"] = gold_answer
            s["predict_answer"] = pred_list[i]["answer"]
            s["max_f1"] = f
            s["id"] = i + 1
            t["paragraphs"].append(s)

    with open(analysis_file, "w", encoding="utf-8") as writer:
        json.dump(t, writer, ensure_ascii=False, sort_keys=False, indent=4, separators=(', ', ': '))
        writer.close()
    print("yes/no/unknown: {}".format(numynu))
    print("split error: {}".format(kk))
    # f1 = F1 / len(gold_list)
    # print("f1: %f" % f1)


if __name__ == '__main__':
    main()
