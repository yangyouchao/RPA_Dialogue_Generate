# 开源主题提取

`topic_schema_zh.json` 从项目根目录用户提供的 `schema.json` 提取 20 个 service/intent 组合。原文件有 26 个服务定义；此次按领域覆盖人工选择 20 个意图，不把同领域的多个服务误算成不同的大主题，也不声称抽取了 20 个领域。

每条保留服务名、意图名、英文描述，并增加中文主题名与描述；不再保存 original_intent 原始意图对象。涵盖账户余额、转账、巴士、日程、活动、航班、公寓、看房、酒店、短租、电影、音乐、租车、餐馆、发型师、景点和天气。筛选清单与翻译在根目录 `extract_open_source.py` 的 `SELECTION` 中，源文件 SHA-256 保存在结果中。

本数据集与 DailyDialog 20 条主题样本统一使用九个字段，顺序为 `id`、`name`、`mode`、`description_en`、`description_zh`、`service_name`、`intent_name`、`seeds`、`boundary`。`seeds` 为字符串数组，其余为字符串；无对应信息的文本字段使用空字符串。DailyDialog 的描述字段保存来源开场及中文译文，服务名和意图名留空；其行号等溯源信息保存在顶层 `source_records`，不混入样本字段。两份数据集的顶层来源元数据按各自来源保留。

文件现在包含一个“开源服务与日常闲聊主题”容器和 40 个小主题，沿用原文件名。前 20 条 SGD 样本的 `seeds` 使用翻译后的意图描述；后 20 条 DailyDialog 样本从开场提炼闲聊种子。所有样本使用相同九个字段，保持原 ID，不嵌套第二个 `subtopics`。`boundary` 是项目接入约束，不属于原文。原始意图中的预约、转账等不是当前无工具脚本能执行的真实操作。

原文件结构与 Google Schema-Guided Dialogue 的服务 schema 一致。可参考官方项目：https://github.com/google-research-datasets/dstc8-schema-guided-dialogue 。官方 SGD 数据集标注 CC BY-SA 4.0；用户本地文件的具体下载版本未知，因此这里以本地源文件和哈希为准确溯源依据，不宣称已验证与官方某版本逐字一致。中文翻译为本次新增内容，使用时保留原数据的署名及适用许可要求。

复现本地提取及合并：`python extract_open_source.py topics`。DailyDialog 输入为 `tmp/dailydialog_20/output/topic_dailydialog_20_zh.json`，必要时先运行 `python tmp/dailydialog_20/build.py` 重建。重复提取重建全部40条，不重复追加。

DailyDialog 的来源元数据完整保存在合并文件顶层 `dailydialog_source`，其 `source_records` 按 `DD_CHAT_*` ID 关联样本；原文及中文改编遵循 CC BY-NC-SA 4.0，署名及链接见该对象。顶层 `source_sha256` 仅指本地 SGD 源文件，不代表合并文件的哈希。旧的50条开场及桌面20条独立文件保留，默认加载合并文件即可使用两类话题。
