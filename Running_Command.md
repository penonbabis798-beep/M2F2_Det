没问题！以下是 M2F2-Det 从**训练、合并、推理到最终评估**的全流程运行指令汇总。你可以直接复制这些指令在终端中依次执行。

### 🟢 阶段一：训练 Deepfake Encoder (显微镜)

用于训练底层的视觉伪造特征提取网络。

```bash
# 启动 Stage 1 训练
bash stage_1_train.sh
# (或者如果你使用的是4卡专用脚本：bash stage_1_run_4gpu.sh)

```

### 🟡 阶段二：训练 Projector (特征对齐)

冻结 LLM，仅训练用于连接视觉编码器和语言大模型的适配器层。

```bash
# 1. 启动 Stage 2 训练
bash stage_2_train.sh

# 2. 训练完成后，将 Stage 2 的 Projector 权重合并到 Vicuna 底座中
python merge_projector.py

```

### 🟠 阶段三：指令微调 (Instruction Tuning)

解冻 LLM（使用 LoRA）和 Projector，让模型学会依据视觉线索回答真伪并给出自然语言解释。

```bash
# 1. 启动 Stage 3 训练 (依赖 Stage 2 合并后的底座)
bash stage_3_train.sh

# 2. 训练完成后，将 Stage 3 产生的 LoRA 权重再次合并，生成最终可用模型
python final_merge.py

```

### 🔵 阶段四：批量推理测试 (Inference)

使用最终合并好的 `M2F2-Det-Final` 模型，对测试集生成判断与解释结果。

```bash
# 1. 运行真伪检测 (Detection)，结果保存为 DDVQA_det_c40.jsonl
bash stage_3_inference_det.sh

# 2. 运行原因解释 (Explanation)，结果保存为 DDVQA_exp_c40.jsonl
bash stage_3_inference_exp.sh

```

### 🟣 阶段五：量化评估 (Evaluation)

计算生成的 jsonl 文件的客观指标分数。

```bash
# 1. 评估真伪检测指标 (Accuracy, F1, Precision, Recall)
python eval/eval_judgement.py --predict_path ./outputs/DDVQA/DDVQA_det_c40.jsonl

# 2. 评估文本解释指标 (CIDEr, ROUGE_L, BLEU, SPICE, METEOR)
python eval/eval_explanation.py \
    --predict_path ./outputs/DDVQA/DDVQA_exp_c40.jsonl \
    --gt_path ./utils/DDVQA_eval/c40/test.jsonl

```

*(注：在执行任何阶段前，请确保你已经通过 `conda activate m2f2` 激活了对应的虚拟环境。)*