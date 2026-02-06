import json
train_ids = set([tuple(x) for x in json.load(open('./utils/FFPP_split/train.json'))])
val_ids = set([tuple(x) for x in json.load(open('./utils/FFPP_split/val.json'))])
intersection = train_ids.intersection(val_ids)
print(f"重叠视频数: {len(intersection)}") # 理论上应为 0