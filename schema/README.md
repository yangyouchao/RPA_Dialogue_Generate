# DailyDialog 中文开场对话（50 条）

## 结果文件

- `dailydialog_50_zh.json`：正式结果，每条仅包含 `id` 和 `dialogue` 两个字段。
- `dailydialog_50_zh.md`：相同内容的阅读版。
- ID 范围：`DD-001` 至 `DD-050`。

## 抽样与提取

- 数据来源：[roskoN/dailydialog](https://huggingface.co/datasets/roskoN/dailydialog)。
- 使用完整 `train` split，共 11,118 条；不使用 validation 或 test。
- 不放回随机抽样 50 条，固定种子 `20260908`，算法为 Python `random.Random(seed).sample(range(11118), 50)`，按抽取顺序编号。
- “第一轮”按开场话语解释：仅取 `utterances[0]`，即每行第一个 `__eou__` 之前的完整发言。发言包含多句话时完整保留，不包含对方回复，也不添加场景摘要。
- 不因开场过短、内容不够具体或原文有瑕疵而重抽，以保持随机样本。

## 人名与翻译

- 移除显性人名及其对应的中文音译，不添加“某某”等姓名占位符。
- 呼语中的姓名直接删除；单纯报告姓名的子句删除；人名担任主语时用代词保留句意。
- 15 条含人名：DD-004、006、009、010、015、016、018、019、020、025、026、032、037、041、042。
- DD-015 将“人名 and I”合并为“我们”；DD-032 删除办公室所属者姓名及接线者的姓名介绍；DD-037 删除“My name is ...”句；DD-042 用“他”保留电话找人的意思。
- 保留妈妈、先生、教授等称谓以及地名、机构名。
- 中文译文逐条生成并核对。只对原文明显的拼写、空格及标点问题做必要的自然化处理。
- DD-007 原文为不规范的“Good coming .”，保守按问候译为“你好”，不补充时段。
- DD-030 原文“yor”按“your”理解。
- DD-046 “Fair-Priced Fares”按票务机构名称译为“平价票务”，不是核实过的官方中文品牌名。
- DD-048 “riding the tiger”结合语境意译为处境危险。
- DD-050 “My Buddha”结合后文拍卖佛像的语境译为“我的佛像”；后续发言仅用于消歧，未写入结果。

## 复现与溯源

- `train.zip`：原始训练集压缩包。官方连接失败后，经用户授权从同一仓库的 [镜像链接](https://hf-mirror.com/datasets/roskoN/dailydialog/resolve/main/train.zip) 下载。
- 已验证 SHA-256 与 [官方文件页面](https://huggingface.co/datasets/roskoN/dailydialog/blob/main/train.zip) 公布值相同：`c5179ab5a9a86a77b9d29114087c6b82cc4cb366abea25a43a0c24be761133f7`。
- `sampling_audit.json`：抽样参数、从 0 开始的原始 train 行号和英文开场原文。此文件用于溯源，保留未经去名的原文，不是正式中文结果。
- `sample_dailydialog.py`：运行后可复现抽样并生成审计文件，不覆盖中文译文。
- `verify_and_render.py`：核对条数、ID、抽样来源、人名清理及译文非空，并由 JSON 生成 Markdown 阅读版。

## 许可与署名

源数据集：Li, Yanran; Su, Hui; Shen, Xiaoyu; Li, Wenjie; Cao, Ziqiang; Niu, Shuzi. 2017. [DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset](https://aclanthology.org/I17-1099/)。Hugging Face 整理版本由 roskoN 发布。

源数据及本次中文改编遵循 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。本次改编包括随机抽样、提取开场、人名清理和中文翻译。
