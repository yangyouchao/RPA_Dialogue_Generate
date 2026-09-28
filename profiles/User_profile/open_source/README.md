# Synthetic-Persona-Chat 用户画像

文件：`User_profile.json`。

- 官方数据集：https://huggingface.co/datasets/google/Synthetic-Persona-Chat
- 固定版本：`a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42`。
- 源文件：`data/Synthetic-Persona-Chat_train.csv`。
- 人物事实取自 `user 1 personas`；`speech_style` 参考同一行 `Best Generated Conversation` 中的 `User 1:` 发言，不使用 `User 2:` 发言。
- 抽取范围：按 CSV 顺序读取 `user 1 personas`，直到去重后得到 50 个用户。本次读取从 0 开始的行号 0 至 124，共 125 条数据记录。行号指 CSV 记录，不是含多行引号文本时的物理行号。
- 将每条画像按逐行事实视为集合；比较前统一英文大小写、标点、常见缩写和同义写法。事实完全相同但顺序不同、一条是另一条的子集，或至少共享 3 条事实时，判为同一用户的变体。传递关联的记录归入同组。
- 每组保留事实数最多的源记录；事实数并列时保留原始行号最早者。不拼接不同记录，不推断新事实。125 条源记录最终保留 50 个用户。
- `persona` 先按原文顺序保留逐句中文翻译，再追加项目创作的现代生活细节；每条去除换行后为 85 至 115 个字符，平均约 95 个字符。扩写不改变原始身份和特征，也不应当作数据集原文引用。
- 扩写中保留了约 20% 的低动力、孤独、回避或消极表达，使 User 画像不会全部呈现积极自我提升倾向；这些倾向属于项目创作内容。
- `source_row_index` 保留被选中记录在原 CSV 中的位置；`id` 按原始行号排序后重新连续编号为 `SPC_001` 至 `SPC_050`。`persona_translations_zh.json` 保存英文句子与人工译文的对应关系，`persona_expansions_zh.json` 按源行号保存创作补充，重复提取时会重新组合两部分。
- 每条记录的 `speech_style` 根据同一 `source_row_index` 的 `Best Generated Conversation` 中 `User 1:` 发言归纳，保存在 `persona_speech_styles_zh.json` 并由提取脚本写入画像。只总结提问、接话和表达方式，不把对话里与画像矛盾的临时说法当成身份事实。源对话是数据集生成的合成对话，并非真人对话；`User 2:` 不用于总结。生成时本轮 `user_behavior` 的语气和篇幅要求优先。

来源署名：Pegah Jandaghi、XiangHai Sheng、Xinyi Bai、Jay Pujara、Hakim Sidahmed；Faithful Persona-based Conversational Dataset Generation with Large Language Models；Google Synthetic-Persona-Chat。官方数据页标注 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本项目另行完成抽取、定位、JSON 转换、中文翻译与创作扩写；创作扩写不属于原数据集内容。

复现：`python extract_open_source.py personas`。它从上述固定版本读取 CSV，检查每个保留行包含 `User 1:` 发言，仅保存画像和风格总结，不保存完整原始对话。也可使用 `--csv 本地CSV路径` 从相同文件提取。

该集合的 `format` 为 `raw_persona`。生成脚本将原始画像交给 User 模拟器，未提供兴趣标签时不进行额外兴趣加权。画像是数据，不是需要执行的操作指令。
