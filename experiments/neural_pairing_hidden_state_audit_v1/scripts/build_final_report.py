"""Build the final evidence report only after every declared study completes."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from linux_runtime import ROOT, save


def read(path): return json.loads((ROOT/path).read_text(encoding='utf8'))


def merge_csv(name, studies):
    rows=[]
    for study in studies:
        p=ROOT/'fresh'/study/name
        if not p.exists():continue
        with p.open(newline='',encoding='utf8') as f:
            rows += [dict(study=study,**r) for r in csv.DictReader(f)]
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    return len(rows)


def main():
    arc=read('fresh/arc_first/ANALYSIS.json');audit=read('fresh/causal_original_shape/MECHANISM_AUDIT.json')
    validation=read('fresh/feedback_validation/ANALYSIS.json');v=validation['groups']['both_receivers']
    assert validation['status']=='complete'
    studies=['causal_pilot_16','causal_original_shape','feedback_validation']
    receipts={s:read('logs/'+('CAUSAL_PILOT_STATE.json' if s=='causal_pilot_16' else s+'_STATE.json')) for s in studies}
    assert all(d['status']=='complete' and d['exit_code']==0 for d in receipts.values())
    causal_seconds=sum(d['gpu_worker_seconds'] for d in receipts.values())
    nrows=merge_csv('intervention_results.csv',studies);nlayers=merge_csv('layerwise_analysis.csv',studies)
    rep_gate=validation['content_specific_replication_gate']
    decision=('通过预定的局部重复门槛，值得研究“保持 Write 范数的发送方状态替换为何能修复部分正确伙伴伤害”；尚未达到完整新机制定论。'
              if rep_gate else '未通过预定的内容特异性重复门槛。当前不应把“反馈污染”包装成已发现的新通用失效机制。')
    report=f'''# nl33w ARC-Challenge Hidden-State Mechanism Audit

完成日期：{datetime.now(timezone.utc).isoformat()[:10]}。本轮按 [最新范围](CURRENT_SCOPE.md) 完成：全量已有评估的逐题重算、16 题探索、95 题原始批量逐层干预，以及全部剩余 16 个正确伙伴伤害事件的验证。**nl20→nl23 重训练保持暂缓；没有新增 Loop、模型结构或大规模训练。**

## 结论

通信损害至少包含两条不同的可干预路径：直接有害的前向写入，以及经过反向通信改变发送方状态后回传的伤害。伙伴最终答对不能保证其每一层发送状态有益；伙伴最终答错也不妨碍其状态提供纠错收益。仅按伙伴最终 argmax 判断“谁可信”会遗漏这两类现象。

{decision}

**没有充分证据把这些效应唯一归因为 Information Bottleneck、Representation Misalignment 或 Downstream Utilization Failure；没有引入循环计算的依据。** 这不是缺少 nl23 权重导致的分析阻塞，而是现有干预对语义内容、状态方向、写入幅度和后续计算的识别仍有限。

## 1. 实验对象与可靠性

官方代码固定 `7e7fcc401a8243f94bca94070db0993d64277874`；最终发布 nl33w 固定 HF revision `29e7ea33878c5bdc2c3cf4e27fffe46b4966859c`。它是最终聊天/私有游戏训练的发布权重，**不是 nl23 QA checkpoint**。基座及 27 个模型文件的固定 revision、SHA/Git blob 校验见 [model manifest](configs/model_transfer.json)。作者 adapter 配置未固定训练时基座 revision，本次下载版本不能冒充其原训练快照。

双方 36 层、hidden 2560；Bridge 层为零基 `[6,9,12,15,18,21,24,27]`。基座和 adapters 为 BF16，Bridge FP32；无量化、无训练。ON/CUT、单向和层干预始终保持同一组 adapters/Bridge 参数。CUT 仅去掉 Write，不调用会改变 adapter 使用方式的 `Pair.freeze`。

原始 QA 数据重建为 train16631 / val595 / test3165，`bench.json` SHA256 `ace6f39f3197037503a2d2f51757ecdf5f3a54101abf850d966e9e7c6f1e97f1`，与作者字节一致。本轮 ARC 仅四选项 1165 题，原始 1172 题中 7 题非四选项。使用作者 QA prompt、answer-prefix 计分、双方 tick/因果 mask，检查伙伴未来 gold letter 不可见。四选项 Margin 定义为 `gold logit − max(wrong logits)`；四选项概率为这四个字母 logits 的 softmax，不是全词表概率。

正确性证据：官方选定 CPU tests **48 passed / 2 skipped**；新增 **12 项** tiny instrumentation 检查通过；真实 BF16 pilot 的 native/CUT 隐藏状态逐值一致。追踪保持原 fused attention；另算注意力诊断，不把显式 softmax 换入实际 forward。

95 题和追加 240 题均恢复原评估的完整 batch、顺序与 padding：全部 ON/CUT 原始四选项 logits **逐值相同**，同形状重复 ON 误差为 0。95 题中保存 38,380 条状态几何量。各条件记录两侧四 logits、四概率与 margin。状态文件保留 native block、pre/post、FP32 raw/applied Write、BF16 effective Write、K/V、gate 与 attention。并非每个反事实都保存完整长序列；同题 full-position donors 在运行时构造，基础状态的指定位置及所有反事实输出已保存。

## 2. 全量 nl33w 行为结果

| 同权重 nl33w，n=1165 | CUT 正确 | ON 正确 | W→C | C→W | 净变化 |
|---|---:|---:|---:|---:|---:|
| Quill | 1010 (86.6953%) | 1003 (86.0944%) | 19 | 26 | −7 / −0.6009 pp |
| Ember | 967 (83.0043%) | 987 (84.7210%) | 42 | 22 | +20 / +1.7167 pp |
| 概率均值 | 1024 (87.8970%) | 1021 (87.6395%) | — | — | −3 / −0.2575 pp |

概率均值是 final adapters 下的同权重组合，不是作者另训的 Solo ensemble。Quill exact McNemar p=.3713、paired bootstrap 95% CI [−1.7167,+0.5150] pp；Ember p=.01686、CI [+0.3433,+3.0901] pp。它们以本 checkpoint/题集为条件，不能覆盖训练种子不确定性。完整统计含作者式保守区间，见 [全量结果](fresh/arc_first/ANALYSIS.json)。

nl33w 自身 CUT 分组为 **902 双对、108 仅 Quill 对、65 仅 Ember 对、90 双错**。108 组中 Quill 被改错11、Ember 被救36；65 组中 Quill 被救10、Ember 被改错8；双对组 Quill/Ember 被改错15/14；双错组被救9/6。

48 个接收方伤害事件中，**21 个伙伴 CUT 和 ON 均答对**。最终错误答案等于伙伴 CUT 错误答案的为12个，等于伙伴 ON 错误答案的为18个。答案相同只能说明行为一致，不能证明复制。61 个纠错事件中，15 个来自双方 CUT 都错；另有15个纠错时伙伴 ON 仍错。结果反驳了把通信作用概括为“复制伙伴最终答案”的充分性，但不自动证明交换了某项具体知识。

全量 [逐题答案变化、概率和 margin](fresh/arc_first/answer_transition_audit.csv) 保留2330行；[nl23/nl33w 对照](checkpoint_separated_question_comparison.csv) 保留1165个独立对齐条目。

## 3. 三阶段因果实验与数值边界

| 阶段 | 样本与选择 | 干预 | 解释范围 |
|---|---|---|---|
| 探索 pilot | 16题，接收方×伤害/纠错×伙伴CUT正误，每格2题，SHA排序 | 每批27条件，8层删除、双单向、幅度、同题 sender/post patch、shuffle/random | 选择依赖已知转移；不是收益估计 |
| 原始形状复核 | 6个事先指定目标及89个原批量邻居，共95题 | 通用27条件；反馈候选所在16题批次另加36个 K/V、gate、逐层反向切断、范数控制条件 | 95题全部精确重放原始 logits；全局63种条件，并非95题每题都跑63种 |
| 独立干预验证 | 16个尚未追踪的正确伙伴伤害目标，8Q/8E；15个原批量含240题 | ON/CUT、目标单向、CUT K/V、CUT gate、CUT K/V+ON范数、重复ON | 穷尽剩余目标，未按新干预效果替换；邻居不是额外选中目标 |

共343个不同问题有因果推理输出，其中335个恢复原始评估批量；16题 pilot 与后续有重叠，**不可相加为351个不同问题**。本轮三个因果 worker 合计 {causal_seconds/60:.2f} 分钟（{causal_seconds/3600:.3f} loaded-worker GPU-hours），含加载和日志，不是独占利用率计费。

关键数值失败：pilot 改为 batch2 时 **5/16 的原始分组改变**。同一 batch 重复完全一致，不能将这种跨形状变化称为随机采样。原始形状95题上，6282个“问题×模型×条件”输出中，BF16 输出与基于同一 BF16 normalized states 的 FP64 head-dot argmax 有29处差异；原始 ON/CUT 中1处。**FP64 dot 仅隔离输出算术，不是全 FP32 backbone。** 原始基准指标仍按 native BF16 报告，所有失败/tie 均保留，没有静默剔除或替换。

## 4. 逐层定位与失败案例

### 4.1 直接有害写入：不是所有伤害都依赖反向反馈

`Mercury_7068635 / Quill`：问电子产品大量使用的不可再生资源，正确选项 gold。CUT margin +0.875，ON −0.750；只保留 Ember→Quill 仍 −0.625。仅删除第15层双向通信变 +0.250；FP64 head-dot 由 −0.7653 变 +0.2017，保持该翻转。其他7个单层删除均未修复。第15层 CUT post patch 为 +0.375，随机范数匹配的第27层 post patch 仍错。证据支持本例的局部有害路径；不能据此指定第15层为全局错误传播层，也不能从 source CUT 错误推出某个 K/V 元素编码了错误答案。

`Mercury_SC_400162 / Ember`：双方 CUT 正确；Ember CUT +2.625、ON −0.500；保留 Quill→Ember 单向仍 −0.625，即使单向发送方恢复为其 CUT 正确状态也未修复。删除第21层变 +0.125；FP64 由 −0.5667 变 +0.0349。它更接近正确发送方状态的直接有害写入，而不是必须先让发送方答错的链。

95题的190个接收方事件中有9伤害/9纠错；8个伤害保持原始转移于 head-dot 诊断。其中7个在单向条件下仍错，1个经反向切断修复。9个纠错都在单向条件下保留。它们是所选批次的描述，不是全1165题的机制发生率。

### 4.2 反馈候选：因果路径成立，内容-only解释失败

`NCEOGA_2013_8_57 / Quill`，问马的化石演化，正确 C。双方 CUT 与 ON Ember 均正确；Quill CUT +0.625 → ON −0.125，gold概率 .6464 → .4669。只保留 Ember→Quill：margin +0.125、gold概率 .5293。其 FP64 dot 分别 +0.67047、−0.06916、+0.13323。

将 Ember 的所有层发送 K/V 替为本题 CUT 状态，Quill恢复 +0.125；Gate-only 无变化；**同时保持 Quill 每个 token 每层的 factual ON FP32 Write 范数后，native margin 为0且选错，FP64 margin −0.02343，仍错**。因此“只要更正内容方向、不改变幅度就能修复”在此例被证伪。范数匹配发生在写回BF16之前，不声称BF16有效残差范数精确相同。状态内容和幅度的耦合、BF16上游状态变化、近边界读出仍需区分。

单层反向删除只在第6/15层使 FP64 margin略正，但 native仍为0且选错；没有一个单层 reverse cut 让两种读出同时正确。联合反向切断的效果不能包装为已找到一个稳健单层反馈开关。普通双向第6/12/15/21层删除能在两种读出中修复；这是不同干预。

### 4.3 最后一层回填与随机修复不能证明“后续不会使用信息”

第27层之后不再通信。同题接收方 CUT post state 的全位置回填必然恢复 CUT continuation，是正对照。所有层使用 CUT sender 等于相应单向运行，同样是实现恒等检查。它们不是两份独立机制证据。

候选中第27层 norm-matched 随机 post delta 在 native BF16 为错误 tie，在 FP64 dot 却变正确。这个失败对照说明：小 margin 的一次修复不足以声称“语义正确方向”；不能把任意末态替换的成功归因于下游利用障碍。完整原题、四选项、逐层条件与对照见 [案例明细](CASE_STUDIES.md)。

## 5. 正确伙伴伤害的追加验证

预定的全部16个新目标已保留，native/FP64 head-dot 转移均保留的为 **{v['transitions_surviving_dot64']}/16**。下表分母始终为16；“两种读出修复”同时要求原始转移在 head-dot 下仍存在，避免将原本数值消失的伤害计成修复。

| 干预 | native修复 | 两种读出均修复 |
|---|---:|---:|
| 切断反向通信 | {v['reverse_cut_repairs_native']} | {v['reverse_cut_repairs_both_readouts']} |
| CUT K/V | {v['clean_kv_repairs_native']} | {v['clean_kv_repairs_both_readouts']} |
| CUT Gate | {v['clean_gate_repairs_native']} | {v['clean_gate_repairs_both_readouts']} |
| CUT K/V + factual ON Write范数 | {v['clean_kv_normmatched_repairs_native']} | {v['clean_kv_normmatched_repairs_both_readouts']} |

范数匹配下通过两种读出的病例：{', '.join(v['clean_kv_normmatched_robust_ids']) or '无'}。这类 patch 固定了 Write L2，但仍可改变多维表示、attention、接收方后续状态及跨层反馈；不独立识别“哪项知识被误读”。[验证逐题表](fresh/feedback_validation/validation_cases.csv) 同时保留失败病例、伙伴反事实正误及全部 margin。

唯一native修复是 `Mercury_SC_400590 / Ember`：ON四选项出现错误argmax tie，native margin=0；FP64 head-dot 的原始ON已正确，margin +0.06717。反向切断和norm-matched K/V的FP64 margin分别 +0.24163 / +0.31907，确实提高margin，但不能称为在两种读出下均纠错。

**修复门槛失败不等于没有因果margin效应。** 对全部16个新目标，固定FP32 Write范数的CUT K/V使13个margin上升；native平均Δmargin +0.16406，FP64 dot平均 +0.18251、中位 +0.18367。只看仍保留原始伤害的13个目标，有11个上升，FP64均值 +0.18457。反向切断在全部16个中也有13个margin上升，FP64均值 +0.16503。它支持一个较弱、可继续检验的候选：**反向通信相关的表示变化会侵蚀正确伙伴场景下的接收方margin，但通常不是错误的充分原因。** 这是固定伤害样本上的次要描述分析；选择依赖ON伤害，且尚未与匹配随机方向/无关内容的margin恢复比较，不能由13/16直接得出语义特异性、总体发生率或新机制显著性。[完整margin效应](secondary_margin_effects.json)。

一个重要配置限制：发布 checkpoint 的16个 Gate weight矩阵全0、bias全2，故 visibility为常量 `sigmoid(2)=0.880797`。已直接检查发布 safetensors，见 [参数核验](logs/RELEASED_GATE_PARAMETERS.json)。所以 Gate-only donor不变是该 checkpoint 的结构恒等对照，不能宣称发现了“动态Gate被污染/失灵”，也不能把它作为与 K/V 平等的独立失效来源。**在常量Gate下，全层CUT K/V的接收方结果也等于对应单向运行；两行不是两份独立证据。** 它不代表接收方 null attention/alpha 不起作用。

## 6. 隐藏状态、Write与Attention

95题的 answer-prefix，ON Write/hidden L2：Quill范围 .03372–.11580、中位 .06392；Ember .04225–.13301、中位 .07209。BF16 effective/raw Write范数比分别 .99528–1.00604 和 .99222–1.00822。通信没有普遍在残差舍入中消失；非零和幅度充足仍不等于任务内容正确。

null attention质量中位 Quill 3.19e−8、Ember 7.01e−7；Ember部分头/问题的均值可达 .25。注意力质量、熵、K/V范数、各层 pre/post/native状态变化均为观察性诊断，不能单凭 attention 峰值或 logit lens 宣称传输了答案。全量 [state dynamics](fresh/causal_original_shape/state_dynamics.csv) 与 [逐层CSV](layerwise_analysis.csv) 可复算；图表有 PNG/PDF/可编辑SVG。

## 7. 与官方 nl23 的严格分离对照

| 证据 / checkpoint，均1165题未过滤 | Quill CUT→ON | Quill rescue/damage | Ember CUT→ON | Ember rescue/damage | 自身CUT分组 双对/仅Q/仅E/双错 |
|---|---|---|---|---|---|
| 作者公开 nl23 正误记录 | 1047→1045 | 22/24 | 1014→1040 | 67/41 | 964 / 83 / 50 / 68 |
| 本次真实 nl33w BF16 推理 | 1010→1003 | 19/26 | 967→987 | 42/22 | 902 / 108 / 65 / 90 |

nl23 的83组：Quill损害10/83、Ember救64/83；50组：Quill救15/50、Ember损害29/50。它们没有拿来选择 nl33w 案例，也没有与 nl33w 分子/分母混用。相同测试索引和gold逐题核对；重建数据SHA一致。nl23公开文件只给正误，不保存具体错误字母、概率与隐藏状态，因此不能判断同一题两套checkpoint是否采用同一个错误，更不能将 nl33w 的干预效果当成 nl23 机制复现。

作者其他 QA/传输保存记录的重算另保留，**不是本次新增checkpoint推理**。31项全任务排除包含训练重合/验证重合/测试内重复；本报告比较统一使用用户要求的未过滤1165题口径，没有把3134题过滤结果混入。

## 8. 竞争解释、研究价值与停止判断

| 假设 | 本轮能说什么 | 不能说什么 / 下一项证伪 |
|---|---|---|
| Information Bottleneck | 真实Write非零；双方CUT错仍可被救，通道不普遍失效 | 未用语义已知、必要私有证据逐段置换定位信息丢失，不能断言具体知识缺失 |
| Representation Misalignment | norm固定的K/V donor可以检验表示变化是否有因果作用，验证结果如上 | 它同时改变多层内容与后续计算；尚无特定语义变量的反事实正确值/错误值成对置换 |
| Downstream Utilization Failure | 保存了桥后及后续native层轨迹，可定位状态差异 | 末层CUT回填是恒等正对照；没有证明有用信息已进入却被后续计算丢弃，故不支持额外深度 |
| Negative Transfer / feedback | ON/CUT显示真实损害；方向切断与K/V patch能定位部分reverse路径；另有直接forward伤害 | 仅匹配错误字母不证明复制；伙伴argmax正确不代表所有中间状态正确；不能宣称整个负迁移现象新颖 |
| 数值/形状解释 | 5/16探索分组跨batch改变；head-dot输出29处不同；原形状精确复核已通过 | 未做全FP32backbone与跨硬件复验，尤其小margin结论仍限于本BF16实现 |

四个概念不是自然互斥的全局标签。对同一伤害事件，先过数值参考门槛，再区分“reverse切断能否修复”与“单向仍伤害”，最后对范数匹配K/V、具体语义变量、后续计算逐段做证伪。不能把所有残余错误自动归到H3/H4。

现有工作已经报告负迁移、错误消息覆盖、表示对齐、消息错配/随机控制及传输与利用的分离。补充筛查包含 [Do Latent Channels Actually Communicate?](https://arxiv.org/html/2607.26773v1)、[When Does Latent Communication Pay?](https://arxiv.org/html/2608.04893v2)、[When Upstream Messages Override Correct Answers](https://arxiv.org/html/2609.36855v1)、[Portable Semantics, Private Dialects](https://arxiv.org/html/2609.11365v1) 和 [StateBridge](https://arxiv.org/html/2608.13317v1)。[逐项范围](sources/PRIOR_ART.md) 区分了消息边界审计、固定relay、最终态prefix与本系统双向逐层residual桥。此筛查不证明“从未有人研究反馈”。

值得进一步研究的是：**在伙伴最终仍答对时，哪些跨层反向路径改变发送K/V、侵蚀接收方margin；这种方向效应能否超过匹配随机扰动？** 第5节的13/16正向margin变化值得作为下一项有对照的假设，但本轮预定的稳健纠错门槛失败，不能事后把它改成成功。当前不能宣称已得到语义完备、跨数值配置稳健、跨任务的新机制。

不需要重建 nl20→nl23 就能完成上述因果定位；QA训练权重只在检验训练阶段是否改变该路径时有额外价值。当前不值得仅为补齐严格复现而重启完整训练链。所有已下载数据、620条own-reply准备成果、原始chain和失败/中止成本pilot都保留。失败训练不是科学假设的反证。

## 9. 交付与复算

- [完整原题/失败案例](CASE_STUDIES.md)，[竞争假设](hypothesis_matrix.md)，[后续决策](NEXT_STEPS.md)。
- 根目录 `intervention_results.csv`：{nrows}行，按study明确分开；`layerwise_analysis.csv`：{nlayers}行。各fresh子目录保存不可混淆的原始结果。
- 原始batch JSON、选择清单、`.pt`状态、COMPLETE/worker退出receipt、环境锁、参数与数据哈希完整保留。公共交付包排除模型权重、SSH凭证、transport及下载签名URL。
- [复算说明](REPRODUCE.md)；环境为 Linux aarch64、GB10、Torch2.11.0+cu130、Transformers5.12.0、PEFT0.21.2。归档哈希见交付manifest。

![逐层删除效果](figures/layer_lesion_margins.png)

![反馈拆解](figures/feedback_path_dissection.png)

![桥诊断](figures/bridge_diagnostics.png)

![独立目标的margin与纠错门槛](figures/validation_margin_effects.png)
'''
    (ROOT/'REPORT.md').write_text(report,encoding='utf8')
    case_studies()
    save(ROOT/'STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='bounded_nl33w_audit_complete',
        current_scope='CURRENT_SCOPE.md',scientific_success_criterion_met=False,
        scientific_result='Localized causal pathways; no unique semantic failure mechanism or full-FP32 robustness established',
        training='deferred_by_user; preserved preparations; no training worker active',
        checkpoint='released_final_nl33w',fresh_arc_n=1165,
        studies={s:dict(status=d['status'],exit_code=d['exit_code'],gpu_worker_seconds=d['gpu_worker_seconds']) for s,d in receipts.items()},
        causal_gpu_worker_seconds=causal_seconds,unique_causally_audited_questions=343,exact_original_batch_questions=335,
        content_specific_replication_gate=rep_gate,new_loop_or_architecture=False,
        public_nl23='separate correctness-only comparison; no nl23 checkpoint inference',
        root_intervention_rows=nrows,root_layerwise_rows=nlayers))
    print(json.dumps(dict(report='REPORT.md',rows=nrows,layers=nlayers,causal_gpu_hours=causal_seconds/3600,replication_gate=rep_gate)))


def case_studies():
    text=['# 原始形状案例明细\n\n原题、选项与gold保留。四选项概率完整位于batch JSON/CSV；表中给gold概率。Margin为gold减最大错误logit。所有层为零基。\n']
    comp={r['question_id']:r for r in csv.DictReader((ROOT/'checkpoint_separated_question_comparison.csv').open(encoding='utf8'))}
    for p in sorted((ROOT/'fresh/causal_original_shape').glob('batch_*.json')):
        d=json.loads(p.read_text(encoding='utf8'))
        for i,c in enumerate(d['cases']):
            if not c['selected']:continue
            x=c['target_receiver'];it=c['item'];gold=it['answer']
            text.append(f"\n## {c['question_id']} / {x}\n\n{it['q']}\n")
            text += [f"\n- {'ABCD'[j]}. {option}" for j,option in enumerate(it['choices'])]
            text.append(f"\n\nGold：{'ABCD'[gold]}。原始nl33w分组：{c['category']}。\n\n| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |\n|---|---|---|---:|---:|---:|")
            names=['cut','link','ember_to_quill_only','quill_to_ember_only']+[f'drop_layer_{l}' for l in [6,9,12,15,18,21,24,27]]+[f'clean_post_{x}_{l}' for l in [6,15,27]]+[f'random_post_{x}_27','shuffle','random_sender']
            names += [n for n in ['clean_kv_all_ember','clean_gate_all_ember','clean_kv_normmatched_all_ember'] if n in d['conditions']]
            for name in names:
                z=d['conditions'][name][x][i]
                text.append(f"\n| {name} | {'ABCD'[z['prediction']]} | {'对' if z['correct'] else '错'} | {z['correct_answer_margin']:+.4f} | {z['letter_probabilities'][gold]:.6f} | {z['dot64_margin']:+.6f} |")
            z=comp[c['question_id']]
            text.append('\n\n独立历史对照，官方nl23 correctness-only：'+', '.join(f"{s} CUT {z['nl23_'+s+'_cut_correct']} / ON {z['nl23_'+s+'_on_correct']}" for s in ['quill','ember'])+'。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。\n')
    text.append('\n## 未通过内容/数值对照的案例\n\nNCEOGA候选：固定ON Write范数后K/V donor未修复；随机post patch在FP64读出也能修复，不能把末态修复解释为语义方向特异性。MCAS_2006_9_44的探索分组曾随batch2改变；恢复原批量才重新得到原始转移。pilot所有5个不稳定case都保留在其ANALYSIS.json，没有静默替换。\n')
    text.append('\n## 全部16个新验证目标（含失败）\n\n下表为同一BF16轨迹的FP64 head-dot margin；完整native结果、四概率及伙伴反事实正误见validation_cases.csv。\n\n| 题号 / 接收方 | CUT | ON | reverse cut | CUT K/V | CUT gate | CUT K/V + ON范数 |\n|---|---:|---:|---:|---:|---:|---:|')
    for r in csv.DictReader((ROOT/'fresh/feedback_validation/validation_cases.csv').open(encoding='utf8')):
        vals=[r[k] for k in ['dot64_cut_margin','dot64_on_margin','reverse_cut_dot64_margin','clean_kv_dot64_margin','clean_gate_dot64_margin','clean_kv_normmatched_dot64_margin']]
        text.append('\n| '+r['question_id']+' / '+r['receiver']+' | '+' | '.join(f'{float(v):+.4f}' for v in vals)+' |')
    (ROOT/'CASE_STUDIES.md').write_text(''.join(text)+'\n',encoding='utf8')


if __name__=='__main__':main()
