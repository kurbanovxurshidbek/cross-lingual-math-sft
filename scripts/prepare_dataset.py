"""Rebuild data/train.jsonl, data/train_chat.jsonl and data/test_en.jsonl from the official GSM8K release.
The 1,000 test problems are drawn with random.Random(42).sample(range(1319), 1000) and kept in original order."""
import json, random, urllib.request
BASE = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/"
load = lambda s: [json.loads(l) for l in urllib.request.urlopen(BASE + s).read().decode().splitlines()]
train, test = load("train.jsonl"), load("test.jsonl")
def dump(path, rows):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
dump("data/train.jsonl", [{"question": r["question"], "answer": r["answer"]} for r in train])
dump("data/train_chat.jsonl", [{"messages": [{"role": "user", "content": r["question"]},
                                             {"role": "assistant", "content": r["answer"]}]} for r in train])
idx = sorted(random.Random(42).sample(range(len(test)), 1000))
dump("data/test_en.jsonl", [{"question": test[i]["question"], "answer": test[i]["answer"]} for i in idx])
json.dump(idx, open("data/test_indices_seed42.json", "w"))
print(len(train), "train /", len(idx), "test")
