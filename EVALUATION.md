# Teachermodel 回复评估

`evaluate_dialogues.py` 读取当前项目生成的原生 `dialogue_*.json`，对每轮 Character / teachermodel 已生成的回复独立评分。`task`、`open_chat`、`topic_chat` 保持原有含义。评分直接追加到每段对话 JSON 最后的 `scores` 字段，原有对话字段保留。

## 评分方法与来源

参考 `rolecraft角色扮演评估` 根目录版本的 `prompts/evaluation/system.txt`、`user.txt` 和 `evaluation_rubrics.py`。保留其逐原则调用、1/3/5 分、中文理由、严格 JSON 输出和现代情境适应规则。

根据本项目要求，移除所有参考回复、小模型对比和相对评分设定。system prompt 第 3 条改为 teachermodel 独立评分；user prompt 删除参考回复块。性格、语言、情境和话题深度原有的相对判据改为独立判据，`话题承接` 判据原文保留。实际文本在 `prompts/evaluation/` 和 `evaluation_rubrics.py`，因此这是一套改编评估，分数不能直接等同于原 benchmark 的分数。

三个模式统一评价五个原则，并按原模式分别统计：

| 维度 | 原则 | 评价内容 |
| --- | --- | --- |
| 角色一致性 | 性格特质 | 是否自然、具体地体现核心人格 |
| 角色一致性 | 语言风格 | 是否自然体现角色风格，避免机械堆砌 |
| 角色一致性 | 情境适应 | 是否理解并回应当前情境 |
| 开放域对话任务能力 | 话题承接 | 是否承接上文并提供新的交流角度 |
| 开放域对话任务能力 | 话题深度 | 剥离角色表达后，理解和有效信息是否深入 |

1 表示明显不足，3 表示基本满足，5 表示表现充分；每项具体标准以 rubric 为准。每轮五次请求，每次只评一项，不让模型输出总分。不将生成模式映射为原 benchmark 的情感陪伴、教育辅导或角色扮演游戏。`task` 目前衡量角色表现和通用对话质量，不提供专门的任务完成率。

评估上下文包括文件内嵌的角色 profile、`scene`、当前回复之前的完整公开历史（含本轮 User 输入）和本轮待测回复。不加入后续轮次、User 画像、私有目标、行为配置，也不把原文件的 system 说明用作裁判指令。没有静默截断历史或 profile；超出模型上下文限制时记录失败，应选择上下文容量足够的裁判模型。

## 配置与运行

沿用项目已有依赖。在 `.env` 中另行设置：

```dotenv
EVAL_API_BASE=https://your-provider.example/v1
EVAL_MODEL=your-judge-model
EVAL_API_KEY=your-judge-key
```

`evaluation_config.json` 管理裁判请求参数，默认 temperature=0、max_tokens=2000、超时 90 秒、失败重试 2 次、请求最小间隔 1 秒。接口使用兼容 Chat Completions 的 HTTP API；不支持 JSON mode 的服务可设置 `judge.json_mode=false`，本地仍严格校验 JSON、原则键、1/3/5 整数及非空中文理由（程序检查非空字符串，不做语言识别）。现有进程环境变量优先于 `.env`。评估不读取生成配置中的旧 `judge` 别名，不需要配置 User/Character API。

先做离线检查。`--input` 指向直接包含 `dialogue_*.json` 的具体批次目录，也可指定单个原生对话 JSON；不递归合并多个批次，不读取训练导出的 `training.jsonl`。

```powershell
python evaluate_dialogues.py --input dialogues/batch_001 --check
python evaluate_dialogues.py --input dialogues/batch_001 --limit 1 --check --preview
```

`--check` 不调用 API、不加载 API 环境变量、不写文件。`--preview` 单独使用仍会执行评分，纯预览请搭配 `--check`。

正式评估及续跑：

```powershell
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/method_a
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/method_a --resume
```

小规模试跑或按模式筛选：

```powershell
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/trial --limit 2
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/chat_only --mode open_chat --mode topic_chat
```

`--limit` 指 Character 回复数，不是对话段数；按文件名和轮次顺序取前 N 条，仅用于试跑，不是随机代表性抽样。2 条回复产生 10 次评分调用，重试另计。默认评价全部轮次。

如果存在配套 `.progress.log`，仅评价状态为 `completed` / `accepted` 的对话；未完成项列入跳过清单。独立 JSON 没有状态日志时，要求完整交替的 User/Character 对话对；不能据此确认整段生成是否完成。角色信息取自原生 JSON 中的 profile 快照，无需重新读取原角色和 User 素材文件。缺少模式或 profile 的旧格式会报错，避免猜测并混入错误统计。

## 结果与比较

每段对话 JSON 的最后新增如下结构。`round_01` 对应第一轮 Character 回复，依此类推；每个评分对象仅包含 `dimension`（评分维度）、`score`、`reason` 三个字段：

```json
"scores": {
  "round_01": [
    {"dimension": "性格特质", "score": 3, "reason": "该轮评分理由"},
    {"dimension": "语言风格", "score": 5, "reason": "该轮评分理由"}
  ]
}
```

`scores` 之后新增 `average_score`，表示该段对话所有轮次、所有维度已成功评分的算术平均值，保留四位小数。缺失评分不计入分母；未开始评分时为 `null`。每次保存评分自动更新均分，未全部评完时它表示当前有效评分的暂时均分。

示例省略其余三个维度和后续轮次。每完成一项即原子保存到对应对话，不再另行生成详细 `scores.json`。续跑直接读取对话末尾的已保存评分，只补评缺失项。评分不会进入后续评估 prompt 或训练导出；重新渲染未改动的对话会保留评分，改变对话内容或场景后重新保存则清除失效评分。

`--output` 现在只指定运行记录目录，其中包含：

- `manifest.json`：评估版本、裁判模型、配置、prompt/rubric 快照与输入指纹。
- `summary.json`：完成情况，整体、按模式和按角色的逐项均分与 1/3/5 分布；`dialogue_macro_mean` 先算各段对话均分再平均，避免长对话占更高权重。`by_dialogue` 列出每段对话的 `average_score`、已完成/应完成评分项数和完成状态（按本次选中的轮次统计）。
- `failures.jsonl`：失败尝试记录。续跑只重试未成功的任务；历史失败记录保留用于追踪。

旧结果迁移可离线运行，无需 API 环境变量，也不会调用裁判模型：

```powershell
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/method_a --migrate-scores
```

脚本先验证全部历史评分与当前对话上下文一致，再逐段追加评分。旧 `scores.json` 重命名为 `scores.legacy.json` 备份；迁移后重复执行不会重复追加。使用旧目录 `--resume` 时也会自动迁移。

仅补算已有评分的每段均分和更新汇总，无需调用 API：

```powershell
python evaluate_dialogues.py --input dialogues/batch_001 --output evaluations/method_a --summarize-only
```

退出码 0 表示全部选中任务成功（或离线检查/迁移成功），1 表示有评分任务未完成，2 表示配置或数据错误。中断时已完成项可续跑。续跑要求输入、筛选后的样本、prompt、rubric、裁判模型和配置一致；迁移评分存储位置不影响输入指纹。改动评分设置后，需备份并清除对应对话的旧 `scores`，再使用新运行目录，避免混用不同裁判结果。输入目录和运行记录目录均使用项目现有锁机制，不能并发生成、渲染或评估同一批对话；强制终止留下 `.generation.lock` 时，确认无运行进程后再删除锁。

比较两种生成方法时，分别输出到 `evaluations/method_a` 与 `evaluations/method_b`，固定裁判及评分设置，使用相同的角色、User、主题和模式/语气分布。先确认两份 `summary.json` 的 `complete=true`、跳过数量和模式分布，再比较每段均分、五项分布及按模式结果。`average_score` 是五项等权的描述性统计，没有对不同维度进行额外校准。不同生成方法的后续 User 输入可能不同，因此独立生成的两批结果是各自上下文中的质量比较，不能视为严格同上下文的配对实验；单次 LLM 评分也不能证明方法显著更优，建议抽检理由并在多个批次复核。

离线测试：

```powershell
python -m unittest test_evaluate_dialogues -v
```
