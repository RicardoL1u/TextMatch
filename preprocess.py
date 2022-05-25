import json
from paddlenlp.transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')


pre_dataset = json.load(open('data/mjjd_element.json','r'))
pre_dataset = pre_dataset[:10]


def get_index_of_all_matching(element,temp:list)->list:
    ret = []
    for idx,unit in enumerate(temp):
        if element == unit:
            ret.append(idx)
    return ret

# the output dataset
dataset = []


for data in pre_dataset:
    processed_data = {}
    concat_text = data['fact']
    for k,v in data['attr'].items():
        concat_text += ('[SEP] ' + k + ': ' + ' '.join(v))

    processed_data['text'] = concat_text
    tokenizer_result = tokenizer(concat_text)
    processed_data['input_ids'] = tokenizer_result['input_ids']
    processed_data['token_type_ids'] = tokenizer_result['token_type_ids']
    sep_indexs = get_index_of_all_matching(tokenizer.sep_token_id,processed_data['input_ids'])
    processed_data['arguments_range'] = []

    last_sep = sep_indexs[0]
    for this_sep in sep_indexs[1:]:
        processed_data['arguments_range'].append((last_sep+1,this_sep))
        last_sep = this_sep

    dataset.append(processed_data)

with open('data/processed.json','w') as f:
    json.dump(dataset,f)
    



