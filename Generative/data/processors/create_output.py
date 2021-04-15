import json

CoQA_dev = "CoQA_data/dev.json"
dev_predict_data = "CoQA_output/fusion_mlm_cls.txt"
output_json = "CoQA_output/fusion_mlm_cls.json"
rule_json = "CoQA_output/fusion_mlm_cls_rule.json"


def txt2output(coqa_file, pred_file, output_file):
    with open(coqa_file, "r", encoding="utf-8") as reader:
        source = json.load(reader)
        input_data = source["data"]
        reader.close()
    with open(pred_file, "r", encoding="utf-8") as reader:
        pred_data = [x.strip() for x in reader.readlines()]
        reader.close()

    pred_data_json = {}  # 将原始txt文件转化为dic,便于后续寻找context
    i = 0
    for paragraph in input_data:
        paragraph_text = paragraph["story"]
        for qa_idx, q in enumerate(paragraph["questions"]):
            qas_id = paragraph['id'] + '#' + str(q["turn_id"])
            pred_data_json[qas_id] = pred_data[i]
            i += 1

    output = []
    for x in pred_data_json:
        example = {}
        id = x.split("#")[0]
        turn_id = int(x.split("#")[1])
        example["id"] = id
        example["turn_id"] = turn_id
        example["answer"] = pred_data_json[x]
        output.append(example)

    with open(output_file, "w", encoding="utf-8") as writer:
        json.dump(output, writer, ensure_ascii=False, sort_keys=False, indent=4, separators=(', ', ': '))
        writer.close()


def rule(src_file, pred_file, rule_file, cased=False):
    with open(src_file, "r", encoding="utf-8") as reader:
        source = json.load(reader)
        input_data = source["data"]
        reader.close()
    with open(pred_file, "r", encoding="utf-8") as reader:
        pred_data = [x.strip() for x in reader.readlines()]
        reader.close()

    pred_data_json = {}
    i = 0
    for paragraphs in input_data:
        paragraph_text = paragraphs["story"]
        for qa_idx, q in enumerate(paragraphs["questions"]):
            qas_id = paragraphs['id'] + '#' + str(q["turn_id"])

            output_tokens = pred_data[i].split(" ")
            output_sequence = ' '.join(output_tokens)
            if cased:
                paragraph = paragraph_text
            else:
                paragraph = paragraph_text.lower()
            for j in range(len(output_tokens)):
                if output_tokens[j] in [",", "-", "—", "'", ".", "/", ":"]:
                    if 0 < j < len(output_tokens) - 1:
                        token_l = output_tokens[j - 1] + output_tokens[j]
                        token_r = output_tokens[j] + output_tokens[j + 1]
                        token_l_r = output_tokens[j - 1] + output_tokens[j] + output_tokens[j + 1]
                        if paragraph.find(token_l_r) != -1:
                            raw_l_r = output_tokens[j - 1] + " " + output_tokens[j] + " " + output_tokens[j + 1]
                            output_sequence = output_sequence.replace(raw_l_r, token_l_r)
                        elif paragraph.find(token_l) != -1:
                            raw_l = output_tokens[j - 1] + " " + output_tokens[j]
                            output_sequence = output_sequence.replace(raw_l, token_l)
                        elif paragraph.find(token_r) != -1:
                            raw_r = output_tokens[j] + " " + output_tokens[j + 1]
                            output_sequence = output_sequence.replace(raw_r, token_r)
                    elif j == 0 and j < len(output_tokens) - 1:
                        token_r = output_tokens[j] + output_tokens[j + 1]
                        if paragraph.find(token_r) != -1:
                            raw_r = output_tokens[j] + " " + output_tokens[j + 1]
                            output_sequence = output_sequence.replace(raw_r, token_r)
                    elif j > 0 and j == len(output_tokens) - 1 and output_tokens[j] != ".":
                        token_l = output_tokens[j - 1] + output_tokens[j]
                        if paragraph.find(token_l) != -1:
                            raw_l = output_tokens[j - 1] + " " + output_tokens[j]
                            output_sequence = output_sequence.replace(raw_l, token_l)
            pred_data_json[qas_id] = output_sequence
            i += 1

    output = []
    for x in pred_data_json:
        example = {}
        id = x.split("#")[0]
        turn_id = int(x.split("#")[1])
        example["id"] = id
        example["turn_id"] = turn_id
        example["answer"] = pred_data_json[x]
        output.append(example)

    with open(rule_file, "w", encoding="utf-8") as writer:
        json.dump(output, writer, ensure_ascii=False, sort_keys=False, indent=4, separators=(', ', ': '))
        writer.close()


def main():
    txt2output(CoQA_dev, dev_predict_data, output_json)
    rule(CoQA_dev, dev_predict_data, rule_json, cased=False)


if __name__ == "__main__":
    main()
