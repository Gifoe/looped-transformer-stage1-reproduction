# R27 执行合同：有限功能来源准入与末loop K/V重放

本文件新增执行细则，不修改已发布proposal v2、CPU授权、输入或阈值。当前仅允许Sol实现/CPU合成；Root源码审核、public freeze及readback后，仍须单独明确START才可运行GPU。不是ARC方法或新增benchmark验证，不训练、不拟合gate、不读ARC outcome/test/其他数据集。

## 1. 不可变输入与范围

绑定proposal v2 SHA256 `5f2b13c148c58ce38a34196a6cccd04e0204ea2056b64bd98401fd36a1791df7`，CPU结构授权，以及manifest `c7e59ff12fc7b70fe74e16b1f49361cd96f390066f168a19f3c9d6cb9cdf942a` 和 `results/r27_ROOT_STRUCTURE_VERIFICATION.json`。Root已独立重建64nonce、全部8prompt/label/offset；两表input长度分别1177/1174。模型、tokenizer、25demo和原template保持冻结。不得重新生成池、改nonce/label、选另一query或template。

按manifest既定8prompt顺序执行；源为每张表四个label token位置，升序且严格位于query之前。四个label ID均来自已冻结单token核验，不给forward附加预期答案。synthetic expected labels仅用于本诊断阶段评价，不得进入hook的mode、强度、位置选择或未来ARC推理。

R26已证明所测同shape内容不变性，但有6/42长度比较不exact。故本轮禁止跨不同长度借donor：同表两个query及其A/B必须同shape/source位置，且实际source跨query exact必须另验。R26不是本轮bitwise独立的替代收据。

## 2. 固定forward计划：最多33，合法Phase A提前结束为9

新增第0次：manifest首A的**no-hook原forward**。读取四depth最后query单位置normalized states及native4全V head。此为事前instrumentation检查+1，不增加样本、候选方法或改变功能门槛。

接着Phase A原8次native4 capture forward，顺序不变。首capture A与第0次的四depth query states、native4全V均须逐bit exact。Phase A全8完整、source query独立和所有完整性通过后，才评价8/8四label子空间映射。若功能门槛失败，不执行Phase B，正常cleanup/参数收尾后报告“完整Phase A未准入”，总9次。不是partial性能分析。

Phase A通过才执行Phase B：原4个B recipient按manifest固定配对顺序，每个依次 Videntity、Vclean、Vshuffle、Kclean、Kshuffle、Kidentity，共24次。最多总33次。不得根据中途哪个K/V较好调整顺序、删条件或增加探针。任何执行/身份/有限性/参数错误均invalid或inconclusive，不允许partial Phase B效果结论。

所有forward使用原native4 `_run(...,use_cache=False)`、原BF16、SDPA、no external mask/position_ids/KV。eval/no_grad，无backprop。只增加本文件明确的诊断hook，不改旧scorer/model/source。native4 head每次固定单位置[1,1,H]，原BF16 LM head->FP32 fullV，避免head batchshape差。

## 3. Hook合同与调用时钟

仅在新r27文件实现。对24 physical layers的 `k_proj`、`v_proj`共48个projection安装forward hook。K位置是projection输出、RoPE前；V同原projection输出，形状[1,S,num_KV_heads*head_dim]，从实际config核验，不假定等于hidden_size。

每个projection每次原native4 forward必须恰好调用4次。固定事件顺序按原源码是每loop逐layer的K再V；总192事件。记录/验证(loop_index,layer,kind)，loop_index从0起，目标仅3。每次forward前清零计数和flags，前次verify未完成不得切mode或重置计数。调用遗漏、多余、乱序、序列shape/source越界均失败，不以局部计数勉强判定末loop。

capture mode在所有调用中返回None、不修改输出；仅loop3对指定四source rows做detach().clone()保存K/V donor。无inplace写原tensor，不读query答案、模型预测或任何ARC label。

patch mode在loop0/1/2以及非目标kind返回None。目标kind、loop3：`patched=output.clone()`，只替换四source rows；其他行逐元素exact保持，dtype/device/shape不变。K、V分开条件执行，不同时patch，不写Q、embedding、MLP、attention output、query residual或LM head。donor来自对应表/版本的第一query，第二query不得另挑donor。

clean是原source位置对应的A donor；identity是对应B donor；shuffle是相同A donor向量多重集按固定四位置循环移位1位。冻结为`index=[1,2,3,0]`：目标第i行接收源第(i+1)%4行。每层、每kind分别同一规则，不能重抽或按label内容改置换。CPU证明多重集保持，真实运行保存donor及映射。

每次forward收集“非目标行完全不变”等GPU布尔标量，forward结束一次聚合核验，避免每layer逐项CPU同步影响预算；覆盖不能减少。identity exact还须独立比较输出state/head，不能只因写入值相等便宣称通过。异常路径finally移除所有已装handles，保留原异常而不被cleanup异常遮掉。

首no-hook forward前验证无既有hook。48个hook全部安装完成后新建原scorer；期间不forward。每次mode/donor/closure状态变更后scorer.invalidate()，即使module hook列表未变也不能绕fingerprint规则。全部卸载后再次invalidate并核全模型hook inventory恢复原空集合。不可修改旧 `_fingerprint` 或原 `_run`。

## 4. Phase A的来源独立性与功能门槛

对2表×2版本×24layers×2种K/V=192对，比较同表两个query的loop4四source rows完整BF16向量，逐bit exact。这个比较不依赖输出标签。不能把只有最后query state接近当source独立，不设容差、不归因length后继续。

完整8次capture、192source pairs、首nohook/capture parity、调用计数和有限性全部通过，才读取synthetic expected label评价。每个prompt的正确label logit严格高于其他3label，8/8才准入Phase B。用FP32 fullV logits的次序判断等价于同一softmax下排序，不加eps。保存完整四label分数及FP64 fullV softmax下四label总质量；质量只描述，不新增阈值。准确措辞为“限定四label子空间对应关系辨别”，不是全V自由生成正确率。

source不独立、p0 parity错误等属于完整性/前提invalid，不输出功能成功率或V/K效果。若只有8/8规则失败而完整性全过，可以正常声明完整Phase A未准入，保存所有8条功能评价；不挑通过实例做Phase B。

## 5. Phase B检查与唯一主假设

所有24 replay均与对应原B比较前三loop normalized states。为避免只检查query而遗漏source早期改动，取**全序列**前三loop BF16 states逐bit exact；首三loop发生任何差异均invalid。这只影响capture与审计，不改变model forward。native4只需保存指定source donor与最终query读出。对应B前三loop原始state缓存可共享，不必为每次相等输出重复保存，引用须绑定raw bytes/hash及真实equal检查。

每个Videntity、Kidentity都要求四depth query states与native4 fullV exact等于对应B baseline。其余条件允许loop4改变，但前三loop仍必须exact。所有hook只改预定义source并正确cleanup/参数不变是独立前提。

所有Phase B完整、身份/首三loop/有限性/计数检查通过、cleanup及参数收尾通过后一次计算：对配对实例i，`m_X=log p_X(y_A)-log p_X(y_B)`。实现可直接以FP64转换后的native4 logits相减，因公共log-normalizer抵消；记录此等价定义，不能用FP32两次softmax造成无谓数值差。

V是唯一主假设：四实例均需 `m_Vclean-m_B>0` 且 `m_Vclean-m_Vshuffle>0`，并至少3/4有 `m_Vclean>0`。strict0，不引入后验容差。K用完全相同规则独立描述；K通过不得替代V失败、不得择优宣称主假设成立。没有ARC rescue/damage/accuracy或R6旧收益结论，成功也不授权ARC部署。

不做未完成Phase B的效应总结。若完整性中途失败，可以停worker节省资源，但只能invalid执行收据，不能输出部分V/K结果。正常9次Phase A未准入与invalid partial必须不同状态。达到Phase A功能gate不等于全实验通过。

## 6. 原始证据与独立审核

保存lossless BF16 donor raw（uint16承载）/state及FP32 fullV arrays，按dtype+shape+rawbytes SHA去重，引用不可仅靠一个exact布尔。至少包括：p0与capture首A四depth query states；全部8 baseline query states/native4 fullV；192source对两侧donor；B baseline前三loop全序列states及每replay的equal/hash绑定；各replay四depth query states/native4 fullV。相同raw可仅存一次，但原始矩阵不能被均值/摘要替代。

记录每forward输入hash、S、context/table/query/version、mode、donor来源、192调用计数/顺序、source mask、未改行检查与前3loop检查；保存已冻结synthetic标签映射，便于Root在完整结果后独立重算。向量容器、RAW、阶段记录及COMPLETE相互SHA绑定。分析器先核schema/完整性/全部raw finite/参数/ledger/阶段数，再进行许可的label评价或最终m计算。

当前loaded model module源、installedSDPA/masking、原config/tokenizer/runtime及scorer源均绑定hash；前后269组参数hash一致。runtime预定depth4不是config文件修改。保存实际模块版本和可发布hash，不上传PRIVATE第三方源码/权重/tokenizer文件。无实际SDPA interception则准确标明source-bound证据，不能误报截获了内部调用。

异常时尽力保存cleanup/参数后状态；缺最终收据就不合格，不能假定参数未改。只有完整合法9或33次状态能进入对应结论，其他数量不冒充正常功能提前停止。

## 7. 预算、停止与实施前合成

唯一job hard180、soft165、SIGKILL179、最后1秒reap，runtime reserve5。soft时停止开始forward并SIGTERM；模型加载、CPU核验、capture、hash/IO、cleanup与退出全计worker实际wall，计入第16ledger。继承15ledger各used_seconds一次：18874.135196831077，余额2725.8648031689227。无retry、额外profile/warmup，不重置cap，实际overrun仍全计费且不解封机制结论。

CPU合成/静态检查至少覆盖：48模块×4calls顺序；capture returnNone与input不变；仅loop3指定kind的4rows替换、other rows exact；固定permutation多重集；identity与跨querydonor共享；configure前verify；安装失败/forward异常时全cleanup；scorer invalidate调用顺序；9与33阶段计数；192source pairs的穷尽唯一性；Phase A invalid与功能不准入区分；Phase B缺条件拒绝partial；V/K不能择优；raw BF16解码与FP32全V独立数值重算；timeout/账本检查。

冻结前Sol提交源码/CPU收据/精确forward计划；Root与Astra静态审查。public readback不是模型执行授权，必须另有明确START。本轮没有实际功能forward。
