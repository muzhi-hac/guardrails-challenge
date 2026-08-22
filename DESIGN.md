# Design

> **Status: outline.** Sections are stubbed; content to follow.

## 1. Domain and why

## 2. Failure modes addressed

## 3. Architecture

### 3.1 Verdicts, not booleans

### 3.2 Tiered cascade

### 3.3 Latency budgets

### 3.4 Configuration-driven client profiles

## 4. Trade-offs considered

- Speed vs. safety
- Simplicity vs. coverage
- Cost vs. reliability

## 已知局限

**检索与语言：**

- 复合词分解是词典驱动的。未登录复合词退化为整词匹配；词典覆盖当前语料，扩语料需
  同步扩词典。失效方向是保守的 —— 退化为精确匹配，不会产生错误召回。
- 没有派生构词法。分词器能对屈折变化做词典化归约，但不桥接动词到名词的派生 ——
  `gedrosselt`（被限速，动词分词）永远到不了 `Drosselung`（限速，名词）。评测集里
  故意同时保留这两种问法，为的是记录边界在哪，而不是把它藏起来。
- 纯词汇检索对改写型（用客户自己的话提问）查询更弱 —— 这是测出来的，不是假定的：
  5 条已知局限用例里，3 条在 k=5 仍命中，2 条命中在第 1 名，因为真实客户问题通常
  仍带着一个能在分词后存活的领域名词。
- `der 1. Platz` 的序数断句仍会误切，语言规则模块里这条尚未修复。更宽的规则会吞掉
  真实的句子边界。
- 货币识别目前耦合在语言规则里。货币是 region 形状的知识，不是 language 形状的；
  正确归属是 profile 未来的 `region` 字段。
- 复合词分解缓存是有界的，因为分词器不只在索引期跑在固定语料上，还在长期运行的
  聊天进程里跑在用户查询上；无界缓存会随用户输入无限增长。

**中转行为观察：验证于 2026-08-22，仅适用于当时配置的第三方中转端点、账户路由、
SDK 版本和模型标识；不代表官方端点、其他渠道或未来行为。**

- 请求携带非法 effort 值时未收到参数错误，使用有效值时也未观察到可验证的约束效果。
  因此本项目不依赖该字段强制控制 judge effort。
- json-schema format 在该测试范围内未强制返回符合 schema 的结果，而是返回普通文本。
  因此 tier-1 judge 使用同日、同范围内验证过的强制 tool choice。
- 携带 `max_tokens=32` 的请求返回了约 700 字符的文本，`stop_reason` 为
  `"end_turn"`——也就是说该中转端点没有在 32 个 token 处截断。因此「思考耗尽预算、
  正文为空」这条路径无法在该端点上用实时调用复现验证；它的正确性靠构造（协议要求
  `stop_reason` 必填）和单元测试兜底，而不是靠一次真实调用观察到的证据。
- 一个候选模型标识在当时路由下返回无可用渠道；模型对比因此调整为该路由下实际可用的
  两个模型。这是可用性驱动的实验调整，不代表该模型在其他渠道不可用。

**检索 recall（`tests/test_recall.py::test_recall_report`，分母见括号；「命中」按
`doc_id` 判定 —— 命中集合里的任意一个期望文档即算命中，不看命中发生在第几名）：**

```
精确词          (n=16)  @1=0.88  @3=0.94  @5=1.00
局限-derivation (n=1)   @1=0.00  @3=0.00  @5=0.00
局限-paraphrase (n=4)   @1=0.50  @3=0.50  @5=0.75
全部            (n=21)  @1=0.76  @3=0.81  @5=0.90
```

## 5. How we know it works
