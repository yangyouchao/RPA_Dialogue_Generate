# 角色多轮对话生成

使用两个兼容 Chat Completions 的大模型 API，分别模拟 User 和 Character，结合用户画像、启动方式及语气与篇幅配置逐轮生成对话。支持具体场景交流、围绕话题闲聊和无预设话题闲聊，以及抽样计划、断点续跑、追加样本和训练数据导出。

当前只有 User 和 Character 调用模型；场景在本地构建，不调用导演、逐轮裁判或质量检查模型。生成完成的状态为 `completed`，不代表质量审核通过。

## 环境与安装

### 固定 User 与 Character

`--user-id` 从 `--users` 指定的画像文件中选择用户，ID 区分大小写。省略时仍随机抽取。它支持 `validate`、`plan`、`generate`，可以与 `--character` 组合固定双方：

```powershell
python dialogue_generate.py validate --user-id SPC_001 --character lu_xun_profile
python dialogue_generate.py generate --users profiles/User_profile/open_source/User_profile.json --user-id SPC_001 --character lu_xun_profile --schemes schema/open_source --conversation-mode task --count 20 --output dialogues/spc001_luxun
```

固定双方时，task 模式每次抽取不同小主题，`--count` 不得超过所选主题数量。open_chat 模式没有主题维度，固定双方后每次最多生成 1 个样本。ID 不存在时会列出当前画像文件的可用 ID。`--append` 可指定新的 User，仅影响新增样本。`--resume` 默认沿用已保存的 User；若显式指定 `--user-id`，批次所有已有样本都必须与之相符，否则报错。该参数不会修改画像源文件，也不同于控制聊天行为的 `--user-behavior`。

需要 Python 3.10+。在项目根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux 的环境激活命令为 `source .venv/bin/activate`，复制配置使用 `cp .env.example .env`。已有 `.env` 时无需再次复制。

编辑本地 `.env`，填写服务商提供的地址、模型名和密钥：

```dotenv
USER_API_BASE=https://your-provider.example/v1
USER_MODEL=your-user-model
USER_API_KEY=your-user-key

CHARACTER_API_BASE=https://your-provider.example/v1
CHARACTER_MODEL=your-character-model
CHARACTER_API_KEY=your-character-key
```

BASE 应包含服务商要求的路径前缀（如 `/v1`），不包含 `/chat/completions`。配置中的示例值不能直接使用。

脚本默认加载自身目录中的 `.env`，也支持 `--env-file 路径`。已有进程环境变量优先于文件；密钥按字面读取，不展开 `${...}`。无需配置 API 即可执行素材校验、抽样计划和离线测试。

## 快速开始

先校验素材并运行离线测试：

```powershell
python dialogue_generate.py validate
python -m unittest test_dialogue_generate test_sampling_quotas -v
```

配置 API 后，先生成一段对话：

```powershell
python dialogue_generate.py generate --count 1 --output dialogues/trial
```

默认从角色目录抽样。指定一个角色时使用 profile 文件名，可省略 `.json`：

```powershell
python dialogue_generate.py generate --character lu_xun_profile --count 5 --output dialogues/lu_xun
```

先保存计划、稍后执行生成：

```powershell
python dialogue_generate.py plan --count 5 --seed 20260908 --output dialogues/batch_001
python dialogue_generate.py generate --resume --output dialogues/batch_001
```

`plan` 会创建计划文件，但不调用模型。已有批次应使用 `--resume`、`--append` 或新的输出目录，普通生成不会覆盖原批次。

## 目录结构

| 路径 | 内容 |
| --- | --- |
| `dialogue_generate.py` | 主程序、提示词、抽样与对话状态管理 |
| `dialogue_config.json` | 请求参数和环境变量名称，不存放密钥 |
| `sampling_config.json` | 每次新建或追加的 mode、tone、response_length 配额比例 |
| `user_behaviors.json` | User 语气与回应篇幅的 9 个组合预设，独立于用户画像 |
| `.env.example` | API 环境变量模板 |
| `profiles/Character_profile/*.json` | 角色设定 |
| `profiles/User_profile/open_source/User_profile.json` | 默认用户素材：从前50条 Synthetic-Persona-Chat 记录去重后保留的20条中文画像 |
| `profiles/User_profile/generated/User_profile.json` | 可选素材：50 条原创合成中文画像 |
| `schema/open_source/topic_schema_20_zh.json` | 默认主题：20 个服务意图与 20 个 DailyDialog 闲聊主题，共 40 条；沿用原文件名 |
| `schema/generated/topic_*.json` | 可选素材：12 个大主题、96 个子主题、192 条种子 |
| `schema.json` | 服务意图提取的本地源文件，主生成程序不直接读取 |
| `extract_open_source.py` | 用户画像提取及主题合并脚本 |
| `test_dialogue_generate.py` | 使用模拟响应的离线测试 |
| `test_sampling_quotas.py` | 配额、舍入、容量检查和批次恢复的离线测试 |
| `dialogues/` | 本地生成结果，默认不提交 |
| `tmp/` | 本地素材处理脚本、中间数据及临时文件，默认不提交 |

## 批次配额配置

直接编辑项目根目录的 `sampling_config.json`，无需通过命令行设置比例。`validate`、新建和追加时自动读取；`--resume`、`render`、`export` 不读取此文件。

```json
{
  "schema_version": "1.0",
  "conversation_mode_distribution": {
    "task": 0.4,
    "topic_chat": 0.3,
    "open_chat": 0.3
  },
  "tone_distribution": {
    "neutral": 0.4,
    "gentle": 0.2,
    "sharp": 0.4
  },
  "response_length_distribution": {
    "minimal": 0.5,
    "short": 0.4,
    "long": 0.1
  }
}
```

每组保留全部类别，值为 0 到 1 的数字且总和为 1。设为 0 可排除该类别；只生成一种类型时，将它设为 1，其余设为 0。例如只生成无话题闲聊，将 `open_chat` 设为 1、`task` 和 `topic_chat` 设为 0。

按当前配置生成 100 段，会分配 40 段 task、30 段 topic_chat、30 段 open_chat；40 段 neutral、20 段 gentle、40 段 sharp；50 段 minimal、40 段 short、10 段 long。这里的数量是独立对话段数，不是每段内部的聊天轮数。

```powershell
python dialogue_generate.py validate --count 100
python dialogue_generate.py plan --count 100 --output dialogues/quota_batch
python dialogue_generate.py generate --resume --output dialogues/quota_batch
```

也可直接运行 `generate --count 100 --output dialogues/quota_batch`。`validate` 按指定 `count` 检查配置、行为预设覆盖和各模式素材容量，并打印分配结果，不写计划、不调用模型。

分配规则：

1. 分别计算三个字段的整数配额：先向下取整，再按小数余量从大到小补齐名额；并列时按程序中固定类别顺序处理，不受 JSON 字段顺序影响。
2. 按各模式的数量划分组，依据剩余 tone 配额按比例分配；再在“模式 × 语气”组内依据剩余长度配额分配。组顺序和最终对话顺序由 `seed` 打乱。
3. 每个字段的批次总数严格满足取整后的配额；交叉组合尽量平衡，不保证每种组合都出现，也不保证小批次精确满足原始百分比。
4. 在各模式内分别抽取 User、Character 和主题，再固定每段的行为。修改 tone 或长度比例不会改变同种子、同 mode 配额下的素材配对。

配额只约束本次创建的计划。追加时按本次 `--count` 和当前配置另算，不补偿历史批次的比例；生成失败不会重新抽样，已完成子集的分布可能暂时不满足整批配额。续跑使用已保存的具体分配。

素材容量不足时，在写入新计划或调用模型前报错，不自动降低配额或改变比例。固定双方时，open_chat 每批最多 1 段；task、topic_chat 各自最多为主题数量，不同模式允许使用同一组人物和主题。

为兼容已有命令，保留 `--conversation-mode` 和 `--user-behavior`：显式模式覆盖 mode 配额；固定行为 ID 覆盖 tone 与长度配额；`--user-behavior random` 恢复旧的逐段随机行为抽样，不保证 tone 与长度数量。正常使用上述配置时请省略这些参数，覆盖情况会显示在终端并保存到日志。没有新增比例命令行参数。

每段 `.progress.log` 的 `state.sampling_batch` 保存所属批次的 `source_config`、`effective_config`、`overrides`、`target_counts`、`actual_counts`、三字段组合数量 `combinations` 及首尾样本 ID。`actual_counts` 指已分配计划数量，不是已成功完成数量。该批次摘要不传给模型，也不写入训练样本；同批次各日志保存同一摘要，不应将它们再次相加。

## 主题抽样与场景构建

默认根据 `sampling_config.json` 的 mode 配额为每段分配启动方式：

| 模式 | 启动依据 | 是否预设目标 |
| --- | --- | --- |
| `task` | 抽取子主题及一条 `seeds` 情境 | 将情境种子作为初始讨论目标 |
| `topic_chat` | 抽取子主题，只将 `name` 用作初始话题 | 否，`user_goal` 为 `null` |
| `open_chat` | 只抽取 User 与 Character，不加载或抽取主题 | 否，话题、场景触发事件和目标均为 `null` |

`task` 和 `topic_chat` 中，`load_catalog()` 读取 `--schemes` 目录直属的所有 `topic_*.json`，将各文件的 `subtopics` 展开为一个主题列表。不会递归加载其他目录，也不会先抽大主题再抽子主题。

`make_quota_jobs()` 按各模式的配额调用 `make_jobs()`，从“角色 × 子主题 × User”全部组合中等概率、不放回抽取。只有 `task` 再从每个选中子主题的 `seeds` 中随机选一条；`topic_chat` 不使用种子和主题原有的任务边界。

- 同一模式内不会重复完整三元组合；不同模式之间不去重，单个角色、用户或主题可以重复出现。
- 不按兴趣、职业、标签、服务名或意图名筛选和加权，不保证各主题覆盖次数相同。
- `id` 用于主题标识和唯一性校验；`group_id`、`group_name` 记录所属容器，不影响抽样权重。
- `task` 和 `topic_chat` 使用同一主题池，不根据主题文件中的 `mode` 自动分类；该字段与命令行启动模式是两回事。只想讨论特定类别时，应提供只包含目标主题的 `--schemes` 目录。
- 固定种子在输入、顺序和数量相同的条件下复现抽样，不保证远端模型回答一致。

`open_chat` 对“角色 × User”组合不放回抽样，不依赖主题目录；该模式分配到的数量不能超过可用组合数。只有本批实际全部分配为 open_chat 时，整个批次才不加载主题目录。各模式允许在实际聊天中自然转移话题。

`build_scene()` 按以下字段在本地组装场景，不使用画像扩写场景，也不调用模型：

| 场景字段 | `task` | `topic_chat` | `open_chat` |
| --- | --- | --- | --- |
| `public.conversation_mode` | `task` | `topic_chat` | `open_chat` |
| `public.background` | 子主题 `name` | 子主题 `name` | `null` |
| `public.trigger` | 抽中的种子 | `null` | `null` |
| `public.boundary` | 子主题 `boundary` | 空字符串 | 空字符串 |
| `user_goal` | 抽中的种子 | `null` | `null` |

三个模式的 `public.relationship` 均为“初次交流”，`public.facts` 为空列表，`user_private`、`character_private` 为“无”。启动配置保存为 `conversation_start`，包含 `mode`、`topic`、`goal`；续跑沿用快照。

例如，在配置文件中将 open_chat、sharp、minimal 各设为 1，对应组的其他值设为 0，再保存计划：

```powershell
python dialogue_generate.py plan --count 5 --output dialogues/open_sharp
python dialogue_generate.py generate --resume --output dialogues/open_sharp
```

围绕抽中话题闲聊时，将配置中的 topic_chat 设为 1，task 和 open_chat 设为 0，再运行：

```powershell
python dialogue_generate.py generate --count 5 --output dialogues/topic_chat
```

切换为原创合成素材：

```powershell
python dialogue_generate.py validate --users profiles/User_profile/generated/User_profile.json --schemes schema/generated
python dialogue_generate.py generate --count 5 --users profiles/User_profile/generated/User_profile.json --schemes schema/generated --output dialogues/synthetic
```

## User 行为配置

User 分为三部分：`profile` 提供身份、兴趣、知识和已有经历等基本信息；`conversation_start` 决定如何开始交流；`user_behavior` 只控制语气与回应篇幅。新任务传给 User 的画像不包含旧的 `communication_style` 字段，原画像文件仍保留该字段；加载合成画像时不再强制要求它存在。自由文本画像不做自动拆分，提示词要求其中的表达风格服从 `user_behavior`，事实仍以画像为准。

新建或追加样本时，从 `user_behaviors.json` 读取行为预设，根据配额分配的 tone 与 response_length 查找对应预设，每段对话固定一个组合。比例由 `sampling_config.json` 控制，不需要逐一设置 9 个组合的权重，也不再默认固定 neutral_short。

| 语气 | 极短 `minimal` | 简短 `short` | 较长 `long` |
| --- | --- | --- | --- |
| 平常 `neutral` | `neutral_minimal` | `neutral_short` | `neutral_long` |
| 柔和 `gentle` | `gentle_minimal` | `gentle_short` | `gentle_long` |
| 犀利 `sharp` | `sharp_minimal` | `sharp_short` | `sharp_long` |

`neutral` 不刻意安慰或施压；`gentle` 表达柔和但不要求附和；`sharp` 表达直接、犀利，可针对实际回复中的含糊、矛盾或无依据断言追问，不捏造对方说过的话或经历。

`minimal` 允许几个字、短语或不完整句子，如“嗯”“不确定”；`short` 通常一句，必要时补一句；`long` 可以几句话展开。长度是倾向，没有字数硬门槛。任何篇幅都允许本轮没有新增信息，不要求逐条回答、主动推进、反问、致谢或总结，也不为凑篇幅编造背景。

```powershell
python dialogue_generate.py validate --count 100
python dialogue_generate.py plan --count 100 --output dialogues/behavior_trial
python dialogue_generate.py generate --resume --output dialogues/behavior_trial
```

通过 `--user-behaviors 路径` 使用自定义文件，文件结构为 `schema_version: "2.0"` 和非空的 `presets` 数组。每个预设只包含以下三个字段，不接受额外字段：

| 字段 | 允许值及含义 |
| --- | --- |
| `id` | 唯一、非空的预设名；`random` 为保留字 |
| `tone` | `neutral` / `gentle` / `sharp`，表达语气 |
| `response_length` | `minimal` / `short` / `long`，回应篇幅 |

```json
{
  "schema_version": "2.0",
  "presets": [
    {"id": "sharp_minimal", "tone": "sharp", "response_length": "minimal"}
  ]
}
```

配额模式下，每种 tone/length 组合只能有一个预设，且必须覆盖本批非零 tone 配额与非零长度配额的全部组合；缺失或重复时生成前报错。例如自定义文件只包含 sharp_minimal，就应将配额中的 sharp 和 minimal 都设为 1。原有固定 ID / random 命令行覆盖方式仍可用于自定义文件。

行为分配使用由 `--seed` 派生的独立随机序列，修改语气或长度比例不会改变同批次的 User、Character、子主题和种子抽样。配额不增加模型调用，也不改变各模式的本地场景规则和结束条件。

实际选中的预设连同 `schema_version` 保存为任务的 `user_behavior` 快照，并在阅读对话 JSON 的同名顶层字段展示。它只作为结构化数据传入 User 的系统上下文，不直接传给 Character，也不作为元数据或系统规则写入训练导出。

`--resume` 直接使用日志中的行为和启动快照，不读取当前行为文件或重新抽样；修改或移走行为配置文件不会改变已有计划。续跑时不能指定 `--user-behavior`、`--user-behaviors` 或 `--conversation-mode`；`render`、`export` 同样不接受这些选项。追加样本可以另选语气、篇幅和启动方式。

新版 `USER_PROMPT` 已解释三种启动方式、语气与篇幅规则，以及无目标闲聊的结束含义。新任务保存 `prompt_version: "2.0"`。已有任务为 `1.0` 或没有版本时使用本次修改前保留的 `LEGACY_USER_PROMPT`，并沿用原行为快照；没有行为字段的旧任务不补加配置。旧版行为文件不能用于新建任务，需要改成上述 2.0 格式。`CHARACTER_PROMPT` 保持原内容。

这些规则由提示词约束，程序只校验配置和输出结构，不评判实际语气或字数，也不因不符合风格而自动重试。离线测试可以检查配置传递与恢复，实际效果需要真实生成后抽检。

## 对话流程与模型配置

流程为：读取配额 → 分配模式与素材、语气和长度 → 保存完整批次计划 → 本地场景 → User 发言 → Character 回复 → 本地结束判断；未结束则进入下一轮。批次和各轮 API 请求均串行执行。

User 接收自己的画像、启动配置、行为配置、场景、目标（可为 `null`）、角色公开姓名和公开历史。Character 仅接收扮演规则、自己的完整 profile 和双方公开对话历史（含 User 本轮最新发言）；不传入整个 `scene` 或 `private`，也不传入 User 画像、行为配置、启动配置、目标和模式。话题和情境只有在公开发言中被提及时才对 Character 可见。场景仍保存在阅读文件和日志中供人工检查，不作为 Character 的额外上下文；此规则同样适用于续跑和重新导出的训练文件。旧对话已经公开说出的内容仍保留在历史中。双方不会直接获得对方的完整画像。历史消息以当前发言者为视角转换：自己的发言标为 `assistant`，对方标为 `user`。

User 返回 `{"message":"公开发言","goal_completed":false}`，Character 返回纯文本。一轮包含双方各一次发言。程序仅在以下情况停止：

1. User 返回 `goal_completed=true`，并完成本轮 Character 回复。
2. 达到 `MAX_ROUNDS = 8` 个完整轮次。

新版 User 提示词不要求最低轮数，`goal_completed` 表示 User 确实想结束本次交流，兼容没有具体目标的闲聊；简短回复本身不代表结束。代码仍只读取布尔标记和轮数，不做语义判断。重复、Character 告别或拒答不会单独触发停止，也没有最终质检。提示词位于主程序中的 `USER_PROMPT`、`LEGACY_USER_PROMPT` 和 `CHARACTER_PROMPT`。

`dialogue_config.json` 当前配置：

| 参数 | 默认值 |
| --- | --- |
| 请求超时 | 90 秒 |
| 失败重试 | 2 次，即单次请求最多尝试 3 次 |
| 客户端请求最小间隔 | 1 秒 |
| User | temperature 0.7，max_tokens 2400 |
| Character | temperature 0.8，max_tokens 4000 |

旧配置中的 `director`、`judge` 当前不使用，也不要求其环境变量。无失败重试时，N 轮对话共调用 API 2N 次。

接口需支持文本 Chat Completions。User 默认使用 JSON mode；服务不支持 `response_format` 时，可将对应 `json_mode` 设为 false，程序仍通过提示词和本地校验要求 JSON。Character 没有 JSON 校验器，因此不会因配置中的 `json_mode=true` 被强制返回 JSON。不自动适配 Responses API、工具调用或厂商专用推理参数。

## 续跑与追加

运行中断后：

```powershell
python dialogue_generate.py generate --resume --output dialogues/batch_001
```

`--resume` 恢复已有样本，跳过完成项；不会按新传入的 `count`、`seed` 和素材路径重新抽样。用户已发言而角色请求失败时，会从角色阶段继续，不重复追加用户发言。恢复依赖原画像文件及其内容摘要，请保留原资料。

向已有目录新增样本：

```powershell
python dialogue_generate.py generate --append --count 2 --seed 20260914 --output dialogues/batch_001
```

`--count` 表示本次新增数量，默认 5；新 ID 从现有最大编号之后开始。`--append` 允许切换素材、角色、启动方式或行为预设，不检查新组合是否与历史批次重复。`plan --append` 只追加计划。`--append` 和 `--resume` 不能同时使用；追加中断后应使用 `--resume`。

网络、部分 HTTP 错误和模型输出格式错误会有限重试；格式重试会附上校验失败原因。不可重试的 HTTP 错误或重试耗尽会停止批次并保存状态。已开始生成的未完成样本会校验模型和配置一致性；切换模型通常应使用新输出目录。

运行时用 `.generation.lock` 防止同目录并发写入。正常退出自动清理；强制终止后，仅在确认没有进程仍运行时删除残留锁再续跑。远端请求完成但尚未保存时中断，恢复可能重复调用该请求。

## 输出与训练导出

每个样本保存两个文件：

- `dialogue_00001.json`：完整轮次的公开对话、场景及目标，新样本另含 `conversation_start` 和 `user_behavior` 配置快照；闲聊的目标为 `null`。结束后增加 `completed_at`。首条 `system` 展示角色 profile，供人工评分参考，不作为此文件中的新增模型指令回传。
- `dialogue_00001.progress.log`：实际为 JSON，含 `events` 和 `state`，保存阶段、素材路径、调用用量及恢复信息。新计划在 `state.sampling_batch` 中保存批次配额摘要。半轮中断时暂存待回复的 User 消息。

文件先写入临时文件再替换。批次统计打印到终端，不自动创建训练文件。日志含本机路径及场景信息，对话含角色资料；仓库默认忽略运行产物。

重新整理已有阅读文件，不调用 API：

```powershell
python dialogue_generate.py render --output dialogues/batch_001
```

手动导出训练数据：

```powershell
python dialogue_generate.py export --output dialogues/batch_001
```

输出为 `training.jsonl`，包含 `completed` 和历史 `accepted` 样本，排除其他状态，并按公开对话去重。`completed` 未经质量审核，训练前需自行筛选。系统上下文与 User 消息的 `loss_mask=0`，Character 回复为 `1`；训练程序必须自行实现掩码支持，该文件并非所有平台通用的微调格式。

旧版路径 `profile/`、`User_profile/`、`scheme/` 在原位置不存在时可映射到新目录。旧日志及检查点仍兼容读取，停在旧 `turn_check`、`quality` 阶段的样本按当前本地结束规则恢复，不再请求裁判。

## 数据来源与提取

开源画像和主题的来源、版本、抽取方式及许可记录见：

- [用户画像来源](profiles/User_profile/open_source/README.md)
- [主题来源与合并规则](schema/open_source/README.md)
- [原创用户画像](profiles/User_profile/generated/README.md)
- [原创主题](schema/generated/README.md)

仓库现有元数据分别记录 Synthetic-Persona-Chat 的 CC BY 4.0、SGD 的 CC BY-SA 4.0 和 DailyDialog 的 CC BY-NC-SA 4.0；默认主题同时包含服务意图和 DailyDialog 改编内容，应保留各自来源与署名。角色 profile 的来源和使用范围需单独确认，本项目未为所有代码和数据统一声明开源许可证。

正常生成直接使用已整理的 JSON，不需要重新下载或提取。`extract_open_source.py personas` 会下载固定版本的画像 CSV，也支持 `--csv` 指定本地文件。

`extract_open_source.py topics` 依赖根目录 `schema.json` 以及本地 `tmp/dailydialog_20/output/topic_dailydialog_20_zh.json`。后者属于默认忽略的中间素材，因此仅克隆仓库不足以直接重跑主题提取；使用已整理的 40 条主题文件即可运行生成。

## Git 提交范围

`.gitignore` 默认排除真实 `.env` 及其变体、虚拟环境、缓存、编辑器个人设置、`tmp/`、`dialogues/` 和运行日志等产物；保留 `.env.example`、源码、测试及 `profiles/`、`schema/` 素材。

如需公开少量示例，可将选定结果整理到单独的 `examples/` 目录。自定义 `--output` 目录也应加入本地忽略规则。`.gitignore` 不会移除 Git 已跟踪的文件，提交前可使用 `git status --short` 和 `git diff --cached` 检查实际内容。
