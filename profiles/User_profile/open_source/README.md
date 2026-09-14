# Synthetic-Persona-Chat 用户画像

文件：`User_profile.json`。

- 官方数据集：https://huggingface.co/datasets/google/Synthetic-Persona-Chat
- 固定版本：`a520ad7f999ca7e6dfdc25fed9f5070bf6f87b42`。
- 源文件：`data/Synthetic-Persona-Chat_train.csv`。
- 字段：`user 1 personas`，不混入 User 2 或生成的对话。
- 抽取范围：CSV 表头之后的前 50 条数据记录，即从 0 开始的行号 0 至 49。行号指 CSV 记录，不是含多行引号文本时的物理行号。
- 保留英文原文、原始顺序及重复项；50 条记录中有 49 种不同的完整文本，不进行去重补抽，也不把相似画像合并。
- 未翻译或扩写画像，未推断原文没有的职业、年龄、知识或兴趣。当前请求中的中文翻译仅用于主题。
- `source_row_index` 和 `id` 是抽取时附加的定位字段，`persona` 是源字段的原样文本。

来源署名：Pegah Jandaghi、XiangHai Sheng、Xinyi Bai、Jay Pujara、Hakim Sidahmed；Faithful Persona-based Conversational Dataset Generation with Large Language Models；Google Synthetic-Persona-Chat。官方数据页标注 CC BY 4.0：https://creativecommons.org/licenses/by/4.0/ 。本次改动仅为抽取指定列、添加定位信息和转换 JSON 结构。

复现：`python extract_open_source.py personas`。它从上述固定版本读取 CSV，仅保存所选画像，不保存原始对话。也可使用 `--csv 本地CSV路径` 从相同文件提取。

该集合的 `format` 为 `raw_persona`。生成脚本将原始画像交给 User 模拟器，未提供兴趣标签时不进行额外兴趣加权。画像是数据，不是需要执行的操作指令。
