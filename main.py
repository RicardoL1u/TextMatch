from paddle.io import DataLoader,Dataset
from paddlenlp.transformers import AutoModel,AutoTokenizer
import paddle
from paddle import optimizer
from util import WarmUp_LinearDecay
import numpy as np
import ot
import json
import datetime

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
            'key_pos':item['arguments_range'],
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
        encoder_rep = self.encoder_linear(encoder_rep) # (bsz,seq,dim)



        # (bsz,key_num,dim)

        
class DomTrain(object):
    def __init__(self, train_loader, valid_loader, args):
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.model = DomTrigger(pre_train_dir=args["pre_train_dir"], dropout_rate=args["dropout_rate"])

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': args["weight_decay"]},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.optimizer = optimizer.AdamW(parameters=optimizer_grouped_parameters, learning_rate=args["init_lr"])
        self.schedule = WarmUp_LinearDecay(optimizer=self.optimizer, init_rate=args["init_lr"],
                                           warm_up_steps=args["warm_up_steps"],
                                           decay_steps=args["lr_decay_steps"], min_lr_rate=args["min_lr_rate"])
        self.model.to(device=args["device"])

    def train(self):
        best_em = 0.0
        self.model.train()
        steps = 0
        while True:
            for item in self.train_loader:
                input_ids, input_mask, input_seg, seq_mask, start_seq_label, end_seq_label, span_label, span_mask = \
                    item["input_ids"], item["input_mask"], item["input_seg"], item["seq_mask"], item["start_seq_label"], \
                    item["end_seq_label"], item["span_label"], item["span_mask"]
                self.optimizer.clear_gradients()
                loss = self.model(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_seg=input_seg,
                    seq_mask=seq_mask,
                    start_seq_label=start_seq_label,
                    end_seq_label=end_seq_label,
                    span_label=span_label,
                    span_mask=span_mask
                )
                loss.backward()
                paddle.nn.ClipGradByGlobalNorm(group_name=self.model.parameters(), clip_norm=self.args["clip_norm"])
                self.schedule.step()
                steps += 1
                if steps % self.args["print_interval"] == 0:
                    print("{} || [{}] || loss {:.3f}".format(
                        datetime.datetime.now(), steps, loss.item()
                    ))
                if steps % self.args["eval_interval"] == 0:
                    f, em = self.eval()
                    print("-*- eval F %.3f || EM %.3f -*-" % (f, em))
                    if em > best_em:
                        best_em = em
                        paddle.save(obj=self.model.state_dict(), path=self.args["save_path"])
                        print("current best model checkpoint has been saved successfully in ModelStorage")

    def eval(self):
        self.model.eval()
        y_pred, y_true = [], []
        with paddle.no_grad():
            for item in self.valid_loader:
                input_ids, input_mask, input_seg, span_mask = item["input_ids"], item["input_mask"], item["input_seg"], item["span_mask"]
                y_true.extend(item["triggers"])
                s_seq, e_seq, p_seq = self.model(
                    input_ids=input_ids,
                    input_mask=input_mask,
                    input_seg=input_seg,
                    span_mask=span_mask
                )
                
                s_seq = s_seq.cpu().numpy()
                e_seq = e_seq.cpu().numpy()
                p_seq = p_seq.cpu().numpy()
                for i in range(len(s_seq)):
                    y_pred.append(self.dynamic_search(s_seq[i], e_seq[i], p_seq[i], item["context"][i], item["context_range"][i]))
        self.model.train()
        return self.calculate_f1(y_pred=y_pred, y_true=y_true)

    def dynamic_search(self, s_seq, e_seq, p_seq, context, context_range):
        ans_index = []
        t = context_range.split("-")
        c_start, c_end = int(t[0]), int(t[1])
        # 先找出所有被判别为开始和结束的位置索引
        i_start, i_end = [], []
        for i in range(c_start, c_end):
            if s_seq[i][1] > s_seq[i][0]:
                i_start.append(i)
            if e_seq[i][1] > e_seq[i][0]:
                i_end.append(i)
        # 然后遍历i_end
        cur_end = -1
        for e in i_end:
            s = []
            for i in i_start:
                if e >= i >= cur_end and (e - i) <= self.args["max_trigger_len"]:
                    s.append(i)
            max_s = 0.0
            t = None
            for i in s:
                if p_seq[i, e] > max_s:
                    t = (i, e)
                    max_s = p_seq[i, e]
            cur_end = e
            if t is not None:
                ans_index.append(t)
        out = []
        for item in ans_index:
            out.append(context[item[0] - c_start:item[1] - c_start + 1])
        return out

    @staticmethod
    def calculate_f1(y_pred, y_true):
        exact_match_cnt = 0
        exact_sum_cnt = 0
        char_match_cnt = 0
        char_pred_sum = char_true_sum = 0
        for i in range(len(y_true)):
            x = y_pred[i]
            y = y_true[i].split("&")
            # 这里则是全词级别的匹配
            exact_sum_cnt += len(y)
            for k in x:
                if k in y:
                    exact_match_cnt += 1
            
            # 这里是单字匹配，也就是char级别的
            x = "".join(x)
            y = "".join(y)
            char_pred_sum += len(x)
            char_true_sum += len(y)
            for k in x:
                if k in y:
                    char_match_cnt += 1
        em = exact_match_cnt / exact_sum_cnt
        precision_char = char_match_cnt / char_pred_sum
        recall_char = char_match_cnt / char_true_sum
        f1 = (2 * precision_char * recall_char) / (recall_char + precision_char)
        return (em + f1) / 2, em

if __name__ == "__main__":
    print("Hello RoBERTa Event Extraction.")
    device = "gpu:0" 
    args = {
        "device": device,
        "init_lr": 2e-5,
        "batch_size": 32,
        "weight_decay": 0.01,
        "warm_up_steps": 1000,
        "lr_decay_steps": 4000,
        "max_steps": 5000,
        "min_lr_rate": 1e-9,
        "print_interval": 100,
        "eval_interval": 500,
        "max_len": 512,
        "max_trigger_len": 6,
        "save_path": "ModelStorage/dominant_trigger.pth",
        "pre_train_dir": "bert-wwm-chinese",
        "clip_norm": 0.25,
        "dropout_rate": 0.5,
    }
    paddle.set_device('gpu:0')
    with open("DataSet/process.p", "rb") as f:
        x = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained("bert-wwm-chinese")
    train_dataset = DomDataset(data=x["train_dominant_trigger_items"], tokenizer=tokenizer, max_len=args["max_len"])
    valid_dataset = DomDataset(data=x["valid_dominant_trigger_items"], tokenizer=tokenizer, max_len=args["max_len"])

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=False, num_workers=4)

    m = DomTrain(train_loader, valid_loader, args)
    m.train()

    
