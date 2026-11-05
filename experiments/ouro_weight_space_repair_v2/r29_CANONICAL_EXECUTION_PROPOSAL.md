# R29 canonical prefix 执行提案

状态：CPU/source-only proposal，尚非新方法、冻结合同或GPU授权。不读取R28部分结果，不改变其运行。工程块64、head块32在效果前固定，不搜索更优形状。

## 判断

**在当前源码路径下数学上可行，值得一个有限、execution-only的位级验收；尚不能宣称批量路径已经等于定义的prefix函数。** 固定shape可能摊薄每token重跑成本；它与R28 true-prefix不是数值同一协议，不能用旧缓存/旧baseline替代匹配重算，不能把primitive变化记作方法提升。

只读核查了pinned Ouro source（历史SHA c5c68fbb368ce2909c257ae2afc50719be8c91539333d3295e19312c4316f413）、config、ouro_cached_scoring.py，以及R26实际SDPA/masking source excerpts。RMSNorm仅沿hidden维求均值；MLP/线性/激活/残差逐token；固定位置RoPE的config rope_scaling=null。唯一跨token值依赖为causal attention，eval dropout=0，当前均full_attention。scorer._run(use_cache=False)直接调用model.model；其中固定range(total_ut_steps)遍历，不依gate结果break。early_exit_gate虽计算，却只作为返回附属列表；外层CausalLM的threshold/weightedexit分支不在本路径。没有发现当前路径中的内容依赖全局退出或跨token非因果归一化。

默认RoPE仍须在运行时核config/source，不得将该结论泛化到dynamic rope或别的模型。没有要求更改mask/backend/BF16：filler是普通token，不是传给attention_mask的padding；仍只传input_ids,use_cache=False。

## 可执行函数定义与块批量化

以每个context独立编码的prompt token数P、已消费answer token数t（从0起）定义：b=floor(t/64)，L=P+64*(b+1)，预测位置i=P-1+t。canonical输入为已知prompt+已消费t个token，补固定token ID0到L。0虽也是BOS/EOS，直接model.model没有generation停止逻辑；不依赖真实EOS或候选剩余长度决定L。

模型depth仍Q5、N012及CF4。超64时仅由已知t进入下一块；不裁切、不依据总答案长度选择shape。超过模型支持最大位置则显式不可执行，不静默截断。

head也定义成固定[1,32,2048]：令a=32*floor(t/32)，32个槽对应预测位置P-1+a…P-1+a+31。canonical版本只放入槽中<=t的原生state，>t槽填BF16零，取槽t-a的head输出。每个depth一致，BF16 head→FP32 fullV。后续每位置的投影/softmax归约保持既有单位置调用shape，避免又引入另一层batch-reduction变量。

快速teacher-forced执行每个64块做一次固定L的forward，可将该块实际已提供的候选token填入未来槽，缺少的槽仍填0；不利用候选长度决定L。相应两个32行head块直接使用实际state。此实现读取future仅作为预计算加速手段，其合法性依赖：对每个取出的t，结果与上述只含past的canonical输入和head排列exact相同。只有该等价成立，才能说返回的是prefix函数；不能用理想因果性偷换CUDA位级等价。

对N1/N2，每个native/CF输入shape必须相同，source143位置在query之前。native donor只由相同shape native forward产生；所有layer source V在数学上不依赖随后answer token。将这些donor按固定映射钳制到CF，按loop/层归纳，预测位置仍只依赖过去：每层attention causal、其余row-local，跨loop同位置状态也是先前causal结果。source的“未来recipient”不回流到较早位置。这排除了数学future信息路径，**不证明GPU shape/content运行数值逐位相同**。R26的42个同shape内容对照是有限支持，不能替代新shape/新head/CF检查。

## 最小有限验收草案（先CPU冻结，不执行）

仅选原24 cohort第一题第一candidate的固定Q prompt和N012，保留25demo。连续token轨迹由合法固定ID周期[1,2,3,4]延伸至128，完全人工已知序列、无ARC gold、无质量评价；不是另一个benchmark。CPU核每个context/source/位置可行，不通过就报告反例，不改token池重试。

固定检验t=0…65：首块全部64位置及块边界后2位置，含32行head切换。两种路径：每个t的canonical prefix+filler，以及相同L的快速整块输入。六个模型条件为Q native5、N0 native4、N1 native4/CF4、N2 native4/CF4；CF仍由其配对native生成donor，不读取任何人工nexttoken来估方向。

66*6=396次canonical forward；两个工程块*6=12次快速forward；首块六条件各固定repeat一次=6，总414次。所有head比较从这些states提取，不再模型forward。该有限验收不声称覆盖所有自然输入；随后若采用此primitive，实际评分任务仍须有预定的独立exact验收覆盖，不得在某些不一致位置偷偷回退/混用两协议。人工覆盖若全过，下一步是对固定自然输入的同类验收及完整匹配protocol预算，不直接看ARC收益挑primitive。

每t、每context逐depth比较当前预测state/rawbytes和固定32行head所取fullV；N1/N2另外比较所有24层source donor BF16raw，以及CF时钟/前三loop约束。使用同输入repeat作为确定性前提。由两路径readout独立重算主系数及全部controls，要求所有有限数值及最终fullV输出exact，不放公差、不只比较argmax。保存raw/hash及完整枚举表，无gold/accuracy。若head不exact而state exact，应明确是head批量化未获等价，不可称semantic泄漏；任何state/donor差同样只判这套执行等价未建立。失败不改64/32/filler/backend重试。

当前是成本与合同草案，具体receipt/schema/参数/cleanup/预算监督须先由Root审查并冻结。原R28运行结束前不准备并发模型job。414次大致小于现R28的2371次，但不是按次数保证线性耗时；按旧四候选总成本折算，69套六条件约123秒纯模型路径，真实同步/完整检查/IO可能显著增加。拟议一次最多400秒（soft385/kill399/reap1、reserve5），仅作为待审预算，不额外profile、不补跑。必须用R28实际结束后的第17ledger核剩余额；不得仍拿2669.57当可用余额。没有新6h额度。

## 资源意义与边界

Root报告的全576 CPU计数为53446 forward/209.55M depth-tokens，13417边界通过；本审查没有自行重跑该计数。canonical理论上可把长候选多个token合到一个块，但跨题reference exact输入去重会改变，固定padding也增加短序列工作量，不能只按64倍宣称加速。下一CPU动作应统计同一576已授权输入下的canonical块数、去重键和depth-token数，再与旧图作纯成本比较；不打开heldout/test逐题。

即使验收通过，576+517+299+1172的完整计划仍需重新预算。新primitive必须同时重算native、Fixed/A1、M2/M3、R6及主方法控制；旧R28结果保持原协议的结论，不合并、不重新解释其成败。保存的新native单位置缓存不能默认与canonical兼容。没有KV复用，没有权重/训练/方法变化，也没有借实现选择来筛效果。

当前下一安全行动仅为Root审查这个执行定义、CPU块任务计数与合成证明检查。无须等待一个不可获得的全GPU形式证明；但在任何批量分数用于研究结论之前，必须实际建立所声明范围的same-shape prefix等价。
