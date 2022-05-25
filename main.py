from email.errors import InvalidMultipartContentTransferEncodingDefect
from lib2to3.pgen2 import token
from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import AutoModel,AutoTokenizer
import paddle
import numpy as np
import ot

class DomDataset(Dataset):
    def __init__(self,data,tokenizer:AutoTokenizer,max_len):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        # text = item['text']
        input_ids = item['inputs_ids']
        if len(input_ids) > self.max_len - 1:
            input_ids = input_ids[:self.max_len-1]
        input_ids += [self.tokenizer.sep_token_id]

        input_seg = [0] * self.max_len 
        for idx,pos in enumerate(item['arguments_range'],start=1):
            input_seg[pos[0]:pos[1]] = [idx] * (pos[1]-pos[0])
        
        # drop extra long part
        input_seg = input_seg[:self.max_len]

        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.tokenizer.pad_token_id] * extra
        
        return {
            'input_ids': paddle.to_tensor(input_ids,dtype='int64'),
            'input_sep': paddle.to_tensor(input_seg,dtype='int64'),
            'data_idx':index
        }





class DomTrigger(paddle.nn.Layer):
    def __init__(self,pre_train:str,dropout_rate: float):
        super().__init__()
        self.bert_encoder = AutoModel.from_pretrained(pre_train)
        self.encoder_linear = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=768,out_features=768),
            paddle.nn.Tanh(),
            paddle.nn.Dropout(dropout_rate),
        )
        # self.start_layer = paddle.nn.Linear(in_features=768,out_features=2)
        # self.end_layer = paddle.nn.Linear(in_features=768,out_features=2)
        # span1和span2是span_layer的拆解, 减少计算时的显存占用
        # self.span1_layer = paddle.nn.Linear(in_features=768, out_features=1, bias_attr=False)
        # self.span2_layer = paddle.nn.Linear(in_features=768, out_features=1, bias_attr=False)  
        # self.selfc = paddle.nn.CrossEntropyLoss(weight=paddle.to_tensor([1.0,10.0],dtype='float32'), reduction="none")
        # self.alpha = alpha
        # self.beta = beta
        self.epsilon = 1e-6
    
    def forward(self,input_ids,input_seg):
        bsz, seq_len = input_ids.shape[0], input_ids.shape[1]
        encoder_rep = self.bert_encoder(input_ids=input_ids, token_type_ids=input_seg)[0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)

        

        key_embs = paddle.to_tensor(key_embs)

    
