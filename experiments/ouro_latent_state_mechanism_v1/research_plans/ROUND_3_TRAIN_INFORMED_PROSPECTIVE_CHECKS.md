# ROUND_3 train-informed prospective checks
本文件在完整train的PQR结果揭示后、validation/test PQR效果揭示前新增。三split已计算完成不等于结果未被使用的独立实验；这是项目内前瞻反证检查，不称独立盲测，不修改既有条件、cohort、评分、锚或停止规则。
来源：train coverage SHA93654dd3a4a53312029e2b33f37253043b5c3086a9330e70571fc47c259dd72d；train rows SHA6b473e38db95c8031ef0ba93d83bb604f52e5496afb2fcfbab60060993e8ecc7；train candidate logits SHA6c6cb5a359e4df1ec0cda7feca2c9e87a3a23b9184fc8b5d169f8677780337e8。Astra独立核对1119全题与1094非demo、八角与旧四角、公式与汇总。
记F_pqr(j)=该角原生产logprob(gold)−logprob(非gold j)，p/q/r=1或2分别指切点P/Q/R的K1/K2来源。固定wrong始终为原K1最强错误候选；不得按干预重新选择。主要readout为先对该题全部非gold j等权平均、再对全split题等权平均；固定wrong版本作为同时报告的敏感性，不能替换主要结果。
H3（位置区域层面的比较）：D_r(j)=F_12r(j)−F_21r(j)，r=1,2。在完整validation与完整test分别预期 mean(D_1)>0 且 mean(D_2)>0。任一split任一r均值<=0，否定该分量的前瞻方向，不以合并split挽救。另报告中位数及正/负/零题数；中位数>0且正数>负数是分布层面的加强证据，若失败必须收窄为仅均值陈述。该命题只比较所定义替换的区域效应，不称Q独有信息、语义推理或特定attention通路。
H4（R背景对P效应的调制）：T_q(j)=[F_2q2(j)−F_1q2(j)]−[F_2q1(j)−F_1q1(j)]，q=1,2。在完整validation与完整test分别预期 mean(T_1)<0 且 mean(T_2)<0。任一split任一q均值>=0即否定相应前瞻方向；报告中位数及正/负/零，不宣称所有题方向一致、不宣称P有害或R是特定门控器。
每个命题均报告原生产score定义及BF16候选logit gap的数值敏感性：只在同请求单token候选、相同readout且完整词表logits来源可核验时，共同log-normalizer在精确代数中抵消。原生产分数不改变；BF16 logit差用精确有理数或先逐候选求线性对比再题内平均，零值不得由任意epsilon分类。两版本分歧作为数值限制，不删题、不重设阈值。
所有结果展示全split；另报告自然K1/K2两端无top tie子集和八角均无top tie子集，二者只是敏感性，不代替全cohort。自然rescue/damage/stable-correct/stable-wrong四组全部展示；其成员由自然结果定义，不能据组内比例估计无偏机制中介份额。预测层面保留所有八角的argmax、top ties、每条P/Q单区域边的预测变化/救回/损伤/净变化；不以准确率端点要求替换gap命题。保存所有反例，禁止根据结果改候选、层或分区。
报告两split各自实际效应大小、分位数、个体符号；如给题级bootstrap须明确为描述性不确定性，不能作为筛选显著项或弥补方向失败。上述H3/H4不是机制定位完成；混合状态OOD、区域长度不等、Q/R已含早期prefix信息、固定prefix不读取未来题的结构限制均保留。
