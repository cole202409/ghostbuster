import numpy as np
import dill as pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from utils.featurize import t_featurize_logprobs, score_ngram, normalize
from utils.symbolic import train_trigram, get_exp_featurize
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton, QLabel, QProgressBar
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 加载模型和参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = pickle.load(open("model/model", "rb"))
mu = pickle.load(open("model/mu", "rb"))
sigma = pickle.load(open("model/sigma", "rb"))
with open("model/all_feat_names.txt", encoding="utf-8") as f:
    all_feat_names = [line.strip() for line in f if line.strip()]
with open("model/features.txt", encoding="utf-8") as f:
    valid_feat_names = [line.strip() for line in f if line.strip()]
name2idx = {name: i for i, name in enumerate(all_feat_names)}
selected_idx = [name2idx[name] for name in valid_feat_names]

def compute_token_logprobs(text, model, tokenizer, max_tokens=1024):
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=max_tokens, truncation=True).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(inputs, labels=inputs)
        logits = outputs.logits
    logprobs = torch.log_softmax(logits, dim=-1).squeeze(0).cpu().numpy()
    token_ids = inputs.squeeze(0).cpu().numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    probs = np.array([logprobs[i, tid] for i, tid in enumerate(token_ids)], dtype=np.float64)
    return tokens, probs

def predict(text):
    # n-gram特征
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tok_ids = tokenizer.encode(text, max_length=2047, truncation=True)
    doc = tokenizer.decode(tok_ids).strip()
    trigram_model = train_trigram()
    tri_scores = np.array(score_ngram(doc, trigram_model, tokenizer.encode, n=3, strip_first=False))
    uni_scores = np.array(score_ngram(doc, trigram_model.base, tokenizer.encode, n=1, strip_first=False))
    # 两个模型logprobs
    neo_tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    neo_mdl = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M").to(device)
    gpt2_tok = tokenizer
    gpt2_mdl = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokens, neo_lp = compute_token_logprobs(doc, neo_mdl, neo_tok, max_tokens=2048)
    gpt2_tokens, gpt2_lp = compute_token_logprobs(doc, gpt2_mdl, gpt2_tok, max_tokens=1024)
    L = min(len(tokens), len(gpt2_tokens))
    neo_lp = neo_lp[:L]
    gpt2_lp = gpt2_lp[:L]
    tri_scores = tri_scores[:L]
    uni_scores = uni_scores[:L]
    tokens = tokens[:L]
    # 特征化
    t_feats = t_featurize_logprobs(neo_lp, gpt2_lp, tokens)
    vector_map = {
        "neo-logprobs": neo_lp,
        "j6b-logprobs": gpt2_lp,
        "trigram-logprobs": tri_scores,
        "unigram-logprobs": uni_scores,
    }
    exp_keys = all_feat_names[len(t_feats):]
    exp_featurize = get_exp_featurize(exp_keys, vector_map)
    exp_feats = exp_featurize(doc)
    all_feats = np.concatenate([t_feats, exp_feats])
    all_feats = all_feats[selected_idx]
    X = (all_feats - mu) / sigma
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prob = clf.predict_proba(X.reshape(1, -1))[:, 1][0]
    return prob

class PredictThread(QThread):
    finished = pyqtSignal(float)
    def __init__(self, text):
        super().__init__()
        self.text = text
    def run(self):
        prob = predict(self.text)
        self.finished.emit(prob)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI生成检测")
        self.resize(500, 300)
        layout = QVBoxLayout()
        self.text_edit = QTextEdit(self)
        self.button = QPushButton("检测", self)
        self.progress = QProgressBar(self)
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.result = QLabel("", self)
        self.result.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.button)
        layout.addWidget(self.progress)
        layout.addWidget(self.result)
        self.setLayout(layout)
        self.button.clicked.connect(self.on_detect)

    def on_detect(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            self.result.setText("请输入文本")
            return
        self.result.setText("")
        self.progress.show()
        self.thread = PredictThread(text)
        self.thread.finished.connect(self.on_result)
        self.thread.start()

    def on_result(self, prob):
        self.progress.hide()
        self.result.setText(f"AI生成概率：{prob:.6f}")

if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
