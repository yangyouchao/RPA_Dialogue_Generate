# Synthetic-Persona-Chat 用户画像

文件：`User_profile.json`。

- 官方数据集：https://huggingface.co/datasets/google/Synthetic-Persona-Chat
- 固定版本：`a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42`。
- 源文件：`data/Synthetic-Persona-Chat_train.csv`。
- 字段：`user 1 personas`，不混入 User 2 或生成的对话。
- 抽取范围：CSV 表头之后的前 50 条数据记录，即从 0 开始的行号 0 至 49。行号指 CSV 记录，不是含多行引号文本时的物理行号。
- 将每条画像按逐行事实视为集合；事实完全相同但顺序不同、一条是另一条的子集，或至少共享 3 条事实且 Jaccard 相似度不低于 0.6 时，判为同一用户的变体。传递关联的记录归入同组。
- 每组保留事实数最多的源记录；事实数并列时保留原始行号最早者。不拼接不同记录，不推断新事实。50 条源记录最终保留 20 个用户，原始行号依次为 `2, 3, 4, 5, 8, 12, 15, 16, 19, 21, 22, 26, 31, 32, 33, 35, 37, 39, 42, 44`。
- `persona` 为逐句中文翻译，顺序与原文一致；不扩写画像或推断原文没有的职业、年龄、知识或兴趣。原始英文可按固定版本、CSV 行号及 `user 1 personas` 列核对。
- `source_row_index` 保留被选中记录在原 CSV 中的位置；`id` 按原始行号排序后重新连续编号为 `SPC_001` 至 `SPC_020`。`persona_translations_zh.json` 保存英文句子与人工译文的对应关系，重复句子使用相同译文。

来源署名：Pegah Jandaghi、XiangHai Sheng、Xinyi Bai、Jay Pujara、Hakim Sidahmed；Faithful Persona-based Conversational Dataset Generation with Large Language Models；Google Synthetic-Persona-Chat。官方数据页标注 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本次改动包含抽取指定列、添加定位信息、转换 JSON 结构和中文翻译。

复现：`python extract_open_source.py personas`。它从上述固定版本读取 CSV，仅保存所选画像，不保存原始对话。也可使用 `--csv 本地CSV路径` 从相同文件提取。

该集合的 `format` 为 `raw_persona`。生成脚本将原始画像交给 User 模拟器，未提供兴趣标签时不进行额外兴趣加权。画像是数据，不是需要执行的操作指令。
