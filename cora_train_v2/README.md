# CORA v2：模态内输出组分桶的监督校准训练

这是当前论文的唯一 CORA 实现（v2，模态内校准）：每个优化步先按各模态在训练集中的题目占比抽定一个模态，整组 256 道题全部来自该模态；对该组前向计算，按本次最大候选概率分桶，统计每桶本次所选答案的正确率，再让桶内每题的 confidence 对齐该正确率，通过平方损失反向传播更新模型参数。更新后下一组重新前向、重新分桶。无 RL、无动作采样、无静态 SFT 分桶目标、无推理后查表。

主配置沿用 SFT 的可训练范围：冻结预训练主干权重，更新语言 LoRA、视觉 merger 和决策头；不新增 confidence 头。答案和 accuracy 可以改变。分桶、argmax 身份和正确率目标停止梯度，梯度只通过当前 confidence 回传。少于最低支持数的桶不参与该步 loss，不能承诺这些题目训练后输出不变。

默认：10 等宽桶，每桶最低 20 题，每次输出组 256 道同一模态的不同题（组内禁止混合模态，update 会校验），4 卡先汇总全局统计，再按有效题数归一化梯度。AdamW（无 weight decay），head/LoRA/merger lr=2e-5/5e-6/1e-6，clip=1，最多 100 步，每 10 步验证。没有 KL 或 SFT 混合损失。分组统计包含本组全部题目，符合当前方法定义；泛化依靠独立验证和测试检验。

训练数据为 data/trimodal_full_train.jsonl，与 SFT 使用相同的三模态训练集，共 157,656 条记录、160,608 道题（文本 15,282、图片 98,142、视频 47,184）。按 source-group 哈希（internal_validation_fraction=0.01）固定排除 1,528 题（文本 164、图片 894、视频 470）不参与分桶统计和反传，仅作训练期监控曲线；其余 159,080 题参与训练组抽样。**报告协议：论文一律评测并报告固定预算（100 步）的最终 checkpoint，不做验证选优；代码内部的 best 指针仅供诊断。**最终测试为文本 764、图片 1858、视频 502；训练/监控/测试独立，来源哈希、checkpoint 哈希及 source-group/request 重叠均有检查。不再使用独立文本开发集，holdout 划分由 group 哈希决定、不可配置为训练数据。

在仓库根目录执行：

    python -m cora_train_v2.train --config configs/cora_train_v2/soft_sft_2b.json --check-data
    python -m unittest cora_train_v2.test_training -v

有可用 GPU 后训练：

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 -m cora_train_v2.train --config configs/cora_train_v2/soft_sft_2b.json

断点续训加 --resume；每个验证点保存全部 SFT 增量参数和优化器状态。每步数据抽样由 seed+step 决定，恢复不重抽随机 confidence。每次先无梯度前向完成整组统计，再在相同参数下逐题重算并反传，最后只更新一次。重算会核对答案、confidence 和候选顺序；启用 gradient checkpointing，并关闭 dropout。不跨优化步骤缓存输出。输出目录：

    save-ckpt/experiments/campaign_20261010_v1/runs/cora_modality_2b/

训练结束后评测最终 checkpoint（step-00100）：

    CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --standalone --nproc_per_node=4 -m cora_train_v2.evaluate --config configs/cora_train_v2/soft_sft_2b.json

评测直接使用训练模型的 argmax、最大概率及完整候选分布，逐模态报告 Acc/ECE/NLL/Brier 和二元置信度指标。测试标签只用于计分。训练日志在 logs/train-*.json，内部验证曲线在 logs/validation-*.json；最终 checkpoint 与评测各有完成清单，严禁用旧 lookup 结果填本方法行。

confidence_group_v1 在本发布中只保留 CORA 复用的纯分桶统计、置信度指标和预测行持久化工具；历史 histogram 后处理方法本体不随代码发布，也不是当前 CORA 训练入口。当前目标由训练集上的本次模型输出生成。数据角色由独立 data.py 校验。
