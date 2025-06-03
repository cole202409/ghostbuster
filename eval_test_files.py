import os
import re
import subprocess

def get_true_label(filename):
    if filename.endswith('-negative.txt'):
        return 0  # 人类文本
    elif filename.endswith('-positive.txt'):
        return 1  # AI文本
    else:
        raise ValueError(f"未知的标签类型: {filename}")

def get_pred_label(prob):
    return 1 if prob > 0.5 else 0

test_dir = 'test_files'
files = [f for f in os.listdir(test_dir) if f.endswith('.txt')]

correct = 0
all_count = 0
for fname in files:
    fpath = os.path.join(test_dir, fname)
    # 调用 classify.py 获取概率
    result = subprocess.run([
        'python', 'classify.py', '--file', fpath
    ], capture_output=True, text=True)
    # 提取概率
    match = re.search(r'预测为正例的概率：([0-9.]+)', result.stdout)
    if not match:
        print(f"未找到概率: {fname}")
        continue
    prob = float(match.group(1))
    pred = get_pred_label(prob)
    true = get_true_label(fname)
    if pred == true:
        correct += 1
    all_count += 1
    print(f"{fname}: 概率={prob:.4f}, 预测={'AI' if pred==1 else '人类'}, 实际={'AI' if true==1 else '人类'}")

acc = correct / all_count if all_count > 0 else 0
print(f"\n模型判断正确率: {acc:.4f} ({correct}/{all_count})")
