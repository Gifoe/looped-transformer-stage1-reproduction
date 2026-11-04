# R26 原路径有限 prefix 因果验收合同

状态：仅设计与实现/CPU合成准备。Root审核、冻结公开输入并readback、再明确start前禁止GPU。原R25及其他关闭方法不复活；不是新部署候选，不测accuracy。旧14ledger余2780.9483967456035秒不变。

## 必要性和范围

Sol已只读核服务器installed transformers4.55.4 / torch2.11.0+cu130：SDPA源SHA d16bbf0bc9fdd197cf02b2917959d2d2c17b8c95555432aa27b4cde83a140033，masking源SHA e29afe5c7058506b22c0e35e5efd8bb749dcfc2e79f32643d442b3e59088eb70；config全24层full_attention、dropout0、sliding=null。原Ouro和接口链支持causal计算。旧R22覆盖首预测位置跨候选states，R25覆盖相同输入selected/full读出；未找到本轮所需的同shape后缀替换与真prefix截断配对证据。

因此唯一新信息是：原实际执行路径在两个预先指定位置上，能否分开通过重复性、未来内容不变性、跨长度位级不变性验收。它不检验方法收益，不证明模型全局因果性，也不预设已有未来泄漏。源function/object属性和已有firstprefix证据不能取代此区分，但本次也不重复全576扫描。

## 预模型输入选择与固定五个context

CPU仅使用原R25已冻结24题cohort顺序、各题原candidate顺序和pinned tokenizer。逐项独立encode_pair生成Q与N0/N1/N2；选第一条四个context的target长度都>=3者。选择只依赖token结构，不读gold、历史预测或效果。不满足者记录结构原因；若完整24无合格项，no-admission退出，不换规则。

N0=`Answer:`；N1/N2沿原25demo/template当前question=N/A、Unknown。每个context独立prefix/target，不假定相同token位置。Q的原input为p+t[:-1]，depth5，保留1..5；N0/N1/N2同样构造各自原input，depth4，保留1..4。第五context是R25固定25demo共享前缀+`Question:`的keyinput，depth4，使用原143 occurrence inventory。

Q/N两个cut分别为len(p)−1和len(p)，即首预测位置与消费一个target后的预测位置。key两个cut为inventory第一和最后key_prediction_position（value位置减1）。所有cut必须>0、cut1<cut2<len(input)−1，确有future。CPU必须验证原字符/prompt、token IDs、value与前驱offset、key共享前缀及原target hashes；不根据模型输出重新选输入。

所有sanitized manifest只存本轮需用文本/token IDs、位置、source provenance，不含gold/prediction/correctness。token IDs中的待预测token仅定义实验干预的future区域，不参与任何修复或value选择。源、输入、CPUreceipt和ledger hashes事前冻结。PRIVATE原始第三方源码仅保存hash/provenance，不发布副本。

## 每context恰好6个原forward，共30个

依次执行：

1. A：完整原input，capture=[cut1,cut2]。
2. B1：与A长度完全相同，<=cut1的IDs完全不变，>cut1每ID变为(id+1)%49152，capture同A。
3. B2：同规则，仅>cut2改变，capture同A。
4. A_repeat：原input再次执行，capture同A。
5. C1：原input[:cut1+1]，仅capture原绝对位置cut1。
6. C2：原input[:cut2+1]，仅capture原绝对位置cut2。

每次使用原scorer `_run(...,positions=...,use_cache=False)`；原model.model调用只含input_ids/use_cache=False。不传外部mask、position_ids或past，不改backend/dtype，不加hook/interception、padding、KV或任何activation干预。固定eval/no_grad，不backprop。使用原depth setter及fingerprint规则，不绕guard。总共30次，失败后不增加探针、profile或retry。

ID变换没有fixed point，允许生成特殊token ID但不调用generate、不提前EOS终止、不据token值生成额外mask；本测试干预的是固定shape tensor，不要求替换结果是自然语言。CPU检查每个future ID确实变化、所有prefix ID保持及长度不变。

### 关键比较域

* A vs A_repeat：两cut、各规定depth，全部检查。
* A vs B1：**仅cut1**；A vs B2：**仅cut2**。B1改动了cut1之后的内容，其中包含cut2，其cut2状态可能合法变化，绝不能将它计入内容不变性失败。虽然B1/B2 capture均为两位置以保持相同head前state选择形状，未保护位置不进入验收谓词。
* A vs C1/C2：分别仅对应cut，构成两个长度比较。

先比较normalized BF16 state；每次head都把该单位置state整理为固定[1,1,H]，经原BF16 LM head再FP32，比较完整V。所有head调用形状一致，不把全长head与单位置head的GEMM差异混入测量。不得只比target logit/argmax。记录每对各depth state/fullV的dtype、shape、hash、exact布尔、最大绝对差及不等元素数；不设置事后容差。

相同输入重复用于判别实际数值可复现性，不是“正常舍入幅度”的估计。head后无需新的log_softmax或修复系数；本轮检验原state/readout，不应用R25函数。

## 完成与解释：不把长度差异称为语义泄漏

必须30次全部完成、supervisor正常、输入/源/所有参数hash与调用范围检查通过后，才生成整体验收结论。中途OOM/timeout/非有限/完整性失败：incomplete或invalid，只报告执行事实，不给partial因果结论。原数值不同本身是待观察结果，不作为提前退出/补跑理由。

完整后按顺序判定（每context也保留同结构结果）：

1. 任一A_repeat非exact：`inconclusive_repeat`。后续差异可记录，但不能归因未来内容或长度。
2. 重复全exact而任一受保护A/B比较非exact：`content_invariance_not_established`。说明实际同shape内容不变性验收失败；不自动称已证明语义泄漏，仍可能需要另外定位实现路径。
3. 同shape A/B全exact、任一A/C非exact：`length_bitwise_sensitive`。所测内容不变性通过，跨长度bitwise不变性不通过；数学因果仍可能正确。不能用非zero差异反推读取了未来token，也不能用“只是舍入”宣布严格prefix合同已满足。
4. 所有上述比较exact：`finite_config_prefix_evidence`。只支持这些五context、两cut、depth/dtype/backend的有限验收，不证明所有输入/硬件全局性质，也不产生accuracy收益。

不计算gold、预测、accuracy、rescues、候选排名或方法晋级。任何结果都不复活R25。不改backend进行“对照直到通过”；若需要kernel定位，必须另写有范围和预算的诊断合同，不能自动继续。

## 来源、模型与资源收尾

实际模型对象module路径及源码、installedSDPA/masking/config、tokenizer文件均核hash并与冻结来源绑定；前后所有named_parameters hash一致。动态depth允许在规定5/4之间切换，是原setter调用，不得误把该预定runtime config变化报成文件漂移；原config文件hash不变。只保存可发布的hash/版本/公开URL与测试结果；PRIVATE源码路径或正文不上传。

由于不加interception，证据准确命名为source-bound原_run入口/实际模块来源及配置记录，不伪称截获了每次内部SDPA调用。原已核源码决定dispatch，实验验证其端到端可观察不变性；二者互补。

一次worker hard120秒，soft105秒停止开始新forward并SIGTERM，SIGKILL119秒，预留最后1秒reap；runtime reserve5保留。CPU输入选择、模型加载、全部forward/head/hash及退出都计入第15ledger实际wall。继承14ledger各used_seconds一次，不使用重复total_used_seconds，不reset原21600cap。当前余2780.9483967456035秒；120秒不是必须消耗的额度，也不保证足够，超限实际仍全计费且不出验收结论。无retry、额外warmup或profile；终止后确认无残留worker。

Root先审Sol实现/CPU synthetic receipts，发布冻结锁并独立readback，然后另行明确start。目前仅允许实现与CPU合成。合成须覆盖输入选择无label、五context两cut、30forward计划、B1cut2不得进入验收、未来变换与prefix保持、统一单位置headshape、四类结论优先级、incomplete拒绝partial及预算继承；无需真实model加载。

## 审查结论

该验收有明确旧证据缺口，且不与闭合方法调参混淆，值得按此范围实现供审。主要阻断条件已写明：B1cut2误判、变更原mask/backend、headshape不同、未完成就解释、或将source记录冒称实际interception。任何一项出现都应修正后再冻结。没有新机制成功宣称，没有GPU授权。
