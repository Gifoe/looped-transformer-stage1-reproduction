# 原始形状案例明细

原题、选项与gold保留。四选项概率完整位于batch JSON/CSV；表中给gold概率。Margin为gold减最大错误logit。所有层为零基。

## Mercury_7068635 / quill

Which nonrenewable resource is used extensively in computers and electronics?

- A. oil
- B. gas
- C. gold
- D. lead

Gold：C。原始nl33w分组：harm_partner_cut_wrong。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | C | 对 | +0.8750 | 0.698201 | +0.836805 |
| link | D | 错 | -0.7500 | 0.289359 | -0.765285 |
| ember_to_quill_only | D | 错 | -0.6250 | 0.322322 | -0.699168 |
| quill_to_ember_only | C | 对 | +0.8750 | 0.698201 | +0.836805 |
| drop_layer_6 | D | 错 | -1.0000 | 0.243764 | -1.045386 |
| drop_layer_9 | D | 错 | -0.6250 | 0.315555 | -0.557623 |
| drop_layer_12 | D | 错 | -0.2500 | 0.386878 | -0.241120 |
| drop_layer_15 | C | 对 | +0.2500 | 0.528587 | +0.201719 |
| drop_layer_18 | D | 错 | -0.8750 | 0.283438 | -0.872894 |
| drop_layer_21 | D | 错 | -0.8750 | 0.282561 | -0.953788 |
| drop_layer_24 | D | 错 | -0.6250 | 0.318532 | -0.614927 |
| drop_layer_27 | D | 错 | -0.7500 | 0.298513 | -0.730240 |
| clean_post_quill_6 | D | 错 | -1.0000 | 0.246360 | -0.950415 |
| clean_post_quill_15 | C | 对 | +0.3750 | 0.534139 | +0.351871 |
| clean_post_quill_27 | C | 对 | +0.8750 | 0.698201 | +0.836805 |
| random_post_quill_27 | D | 错 | -0.7500 | 0.307785 | -0.700961 |
| shuffle | D | 错 | -1.3750 | 0.192650 | -1.365913 |
| random_sender | C | 对 | +1.2500 | 0.760674 | +1.356854 |

独立历史对照，官方nl23 correctness-only：quill CUT 1 / ON 0, ember CUT 0 / ON 0。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## Mercury_400811 / quill

To produce light, the atoms inside a light bulb transform electrical energy into what other kind of energy?

- A. heat
- B. chemical
- C. mechanical
- D. electromagnetic

Gold：D。原始nl33w分组：rescue_both_cut_wrong。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | A | 错 | -0.1250 | 0.468595 | -0.008196 |
| link | D | 对 | +0.3750 | 0.592484 | +0.351976 |
| ember_to_quill_only | D | 对 | +0.3750 | 0.592493 | +0.390330 |
| quill_to_ember_only | A | 错 | -0.1250 | 0.468595 | -0.008196 |
| drop_layer_6 | D | 对 | +0.2500 | 0.561952 | +0.210166 |
| drop_layer_9 | D | 对 | +0.2500 | 0.561978 | +0.201347 |
| drop_layer_12 | D | 对 | +0.5000 | 0.622264 | +0.467328 |
| drop_layer_15 | D | 对 | +0.1250 | 0.531027 | +0.041345 |
| drop_layer_18 | D | 对 | +0.5000 | 0.622270 | +0.495997 |
| drop_layer_21 | A | 错 | +0.0000 | 0.499860 | -0.011882 |
| drop_layer_24 | D | 对 | +0.6250 | 0.651096 | +0.602124 |
| drop_layer_27 | D | 对 | +0.6250 | 0.651211 | +0.558446 |
| clean_post_quill_6 | D | 对 | +0.2500 | 0.561945 | +0.231860 |
| clean_post_quill_15 | A | 错 | -0.2500 | 0.437618 | -0.191728 |
| clean_post_quill_27 | A | 错 | -0.1250 | 0.468595 | -0.008196 |
| random_post_quill_27 | D | 对 | +0.5000 | 0.622316 | +0.446409 |
| shuffle | A | 错 | +0.0000 | 0.499512 | -0.023880 |
| random_sender | D | 对 | +0.6250 | 0.651116 | +0.634496 |

独立历史对照，官方nl23 correctness-only：quill CUT 1 / ON 1, ember CUT 1 / ON 1。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## Mercury_SC_400162 / ember

It is important to use recycled paper because it

- A. comes in colors.
- B. has a smoother appearance.
- C. creates less pollution.
- D. reduces the number of trees cut down.

Gold：D。原始nl33w分组：harm_both_cut_correct。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | D | 对 | +2.6250 | 0.932327 | +2.617542 |
| link | C | 错 | -0.5000 | 0.375224 | -0.566739 |
| ember_to_quill_only | D | 对 | +2.6250 | 0.932327 | +2.617542 |
| quill_to_ember_only | C | 错 | -0.6250 | 0.347528 | -0.690411 |
| drop_layer_6 | C | 错 | -0.5000 | 0.375407 | -0.501723 |
| drop_layer_9 | C | 错 | -0.5000 | 0.374917 | -0.442235 |
| drop_layer_12 | C | 错 | -0.7500 | 0.319147 | -0.676323 |
| drop_layer_15 | C | 错 | -0.2500 | 0.435485 | -0.262997 |
| drop_layer_18 | C | 错 | -0.3750 | 0.405660 | -0.340029 |
| drop_layer_21 | D | 对 | +0.1250 | 0.529405 | +0.034919 |
| drop_layer_24 | C | 错 | +0.0000 | 0.498436 | -0.072981 |
| drop_layer_27 | C | 错 | -0.3750 | 0.406586 | -0.314519 |
| clean_post_ember_6 | C | 错 | -0.5000 | 0.375309 | -0.545750 |
| clean_post_ember_15 | C | 错 | -0.3750 | 0.404737 | -0.371942 |
| clean_post_ember_27 | D | 对 | +2.6250 | 0.932327 | +2.617542 |
| random_post_ember_27 | C | 错 | -0.6250 | 0.345257 | -0.697906 |
| shuffle | C | 错 | +0.0000 | 0.497390 | -0.087327 |
| random_sender | D | 对 | +2.2500 | 0.904507 | +2.362182 |

独立历史对照，官方nl23 correctness-only：quill CUT 1 / ON 1, ember CUT 1 / ON 1。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## Mercury_SC_401277 / quill

Some marine organisms living deep in the ocean can produce their own light. The light they produce is most likely used for which of these activities?

- A. moving through the water
- B. producing oxygen
- C. attracting prey
- D. escaping predators

Gold：C。原始nl33w分组：rescue_partner_cut_correct。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | D | 错 | -0.1250 | 0.457783 | -0.149756 |
| link | C | 对 | +1.6250 | 0.728399 | +1.617530 |
| ember_to_quill_only | C | 对 | +1.6250 | 0.728536 | +1.617517 |
| quill_to_ember_only | D | 错 | -0.1250 | 0.457783 | -0.149756 |
| drop_layer_6 | C | 对 | +1.7500 | 0.762753 | +1.656255 |
| drop_layer_9 | C | 对 | +0.7500 | 0.606675 | +0.759257 |
| drop_layer_12 | C | 对 | +1.3750 | 0.663189 | +1.342622 |
| drop_layer_15 | C | 对 | +1.6250 | 0.728536 | +1.712006 |
| drop_layer_18 | C | 对 | +1.8750 | 0.764284 | +1.835286 |
| drop_layer_21 | C | 对 | +1.2500 | 0.673243 | +1.242784 |
| drop_layer_24 | C | 对 | +1.7500 | 0.752412 | +1.828563 |
| drop_layer_27 | C | 对 | +1.6250 | 0.749765 | +1.619464 |
| clean_post_quill_6 | C | 对 | +1.7500 | 0.741027 | +1.785742 |
| clean_post_quill_15 | C | 对 | +0.6250 | 0.576581 | +0.639614 |
| clean_post_quill_27 | D | 错 | -0.1250 | 0.457783 | -0.149756 |
| random_post_quill_27 | C | 对 | +1.6250 | 0.792289 | +1.614767 |
| shuffle | C | 对 | +1.1250 | 0.697723 | +1.132223 |
| random_sender | D | 错 | -1.5000 | 0.178321 | -1.544247 |

独立历史对照，官方nl23 correctness-only：quill CUT 0 / ON 0, ember CUT 1 / ON 0。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## NCEOGA_2013_8_57 / quill

Which would indicate that a series of fossils represent the evolution of the horse?

- A. All of the fossils were found in the same layers of rocks as horse fossils.
- B. All of the fossils are completely identical to horse bones.
- C. All of the fossils show similar structures to that of the modern horse.
- D. All of the fossils were located in the same place on Earth as horses.

Gold：C。原始nl33w分组：harm_both_cut_correct。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | C | 对 | +0.6250 | 0.646414 | +0.670471 |
| link | A | 错 | -0.1250 | 0.466946 | -0.069162 |
| ember_to_quill_only | C | 对 | +0.1250 | 0.529292 | +0.133231 |
| quill_to_ember_only | C | 对 | +0.6250 | 0.646414 | +0.670471 |
| drop_layer_6 | C | 对 | +0.1250 | 0.529445 | +0.185702 |
| drop_layer_9 | A | 错 | -0.6250 | 0.347161 | -0.617587 |
| drop_layer_12 | C | 对 | +0.1250 | 0.529445 | +0.162810 |
| drop_layer_15 | C | 对 | +0.3750 | 0.590807 | +0.312363 |
| drop_layer_18 | A | 错 | -0.2500 | 0.436812 | -0.316908 |
| drop_layer_21 | C | 对 | +0.2500 | 0.559526 | +0.214881 |
| drop_layer_24 | A | 错 | -0.1250 | 0.466425 | -0.084517 |
| drop_layer_27 | A | 错 | +0.0000 | 0.498095 | +0.091202 |
| clean_post_quill_6 | A | 错 | +0.0000 | 0.498301 | -0.053907 |
| clean_post_quill_15 | A | 错 | +0.0000 | 0.497625 | +0.013059 |
| clean_post_quill_27 | C | 对 | +0.6250 | 0.646414 | +0.670471 |
| random_post_quill_27 | A | 错 | +0.0000 | 0.498268 | +0.021898 |
| shuffle | A | 错 | -0.1250 | 0.466424 | -0.059482 |
| random_sender | C | 对 | +1.1250 | 0.744742 | +1.114798 |
| clean_kv_all_ember | C | 对 | +0.1250 | 0.529292 | +0.133231 |
| clean_gate_all_ember | A | 错 | -0.1250 | 0.466946 | -0.069162 |
| clean_kv_normmatched_all_ember | A | 错 | +0.0000 | 0.498212 | -0.023429 |

独立历史对照，官方nl23 correctness-only：quill CUT 1 / ON 0, ember CUT 1 / ON 0。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## MCAS_2006_9_44 / ember

An astronaut drops a 1.0 kg object and a 5.0 kg object on the Moon. Both objects fall a total distance of 2.0 m vertically. Which of the following best describes the objects after they have fallen a distance of 1.0 m?

- A. They have each lost kinetic energy.
- B. They have each gained the same amount of potential energy.
- C. They have each lost the same amount of potential energy.
- D. They have each gained one-half of their maximum kinetic energy.

Gold：D。原始nl33w分组：harm_partner_cut_wrong。

| 条件 | 预测 | 正误 | native margin | gold概率 | FP64-dot margin |
|---|---|---|---:|---:|---:|
| cut | D | 对 | +0.1250 | 0.387899 | +0.036584 |
| link | C | 错 | -0.5000 | 0.200553 | -0.494295 |
| ember_to_quill_only | D | 对 | +0.1250 | 0.387899 | +0.036584 |
| quill_to_ember_only | C | 错 | -0.3750 | 0.233895 | -0.412112 |
| drop_layer_6 | C | 错 | -0.3750 | 0.221346 | -0.448825 |
| drop_layer_9 | C | 错 | -0.5000 | 0.204813 | -0.423759 |
| drop_layer_12 | C | 错 | -0.3750 | 0.221346 | -0.419982 |
| drop_layer_15 | C | 错 | -0.3750 | 0.221346 | -0.417641 |
| drop_layer_18 | C | 错 | -0.5000 | 0.216449 | -0.442860 |
| drop_layer_21 | C | 错 | -0.3750 | 0.225922 | -0.314766 |
| drop_layer_24 | C | 错 | -0.3750 | 0.254624 | -0.403058 |
| drop_layer_27 | C | 错 | -0.2500 | 0.253009 | -0.307523 |
| clean_post_ember_6 | C | 错 | -0.5000 | 0.207674 | -0.445965 |
| clean_post_ember_15 | C | 错 | -0.3750 | 0.216379 | -0.370211 |
| clean_post_ember_27 | D | 对 | +0.1250 | 0.387899 | +0.036584 |
| random_post_ember_27 | C | 错 | -0.5000 | 0.208725 | -0.457946 |
| shuffle | C | 错 | -0.2500 | 0.257100 | -0.239752 |
| random_sender | D | 对 | +0.3750 | 0.408884 | +0.370216 |

独立历史对照，官方nl23 correctness-only：quill CUT 1 / ON 1, ember CUT 0 / ON 1。不据此推断nl23采用相同错误选项或具有相同隐藏状态路径。

## 未通过内容/数值对照的案例

NCEOGA候选：固定ON Write范数后K/V donor未修复；随机post patch在FP64读出也能修复，不能把末态修复解释为语义方向特异性。MCAS_2006_9_44的探索分组曾随batch2改变；恢复原批量才重新得到原始转移。pilot所有5个不稳定case都保留在其ANALYSIS.json，没有静默替换。

## 全部16个新验证目标（含失败）

下表为同一BF16轨迹的FP64 head-dot margin；完整native结果、四概率及伙伴反事实正误见validation_cases.csv。

| 题号 / 接收方 | CUT | ON | reverse cut | CUT K/V | CUT gate | CUT K/V + ON范数 |
|---|---:|---:|---:|---:|---:|---:|
| MCAS_2001_5_3 / ember | +0.4496 | -1.1298 | -0.9468 | -0.9468 | -1.1298 | -0.9937 |
| Mercury_SC_400062 / quill | +0.4714 | -0.3464 | -0.0258 | -0.0258 | -0.3464 | +0.0136 |
| Mercury_7159320 / ember | -0.0472 | -2.0414 | -1.6278 | -1.6278 | -2.0414 | -1.6581 |
| Mercury_7213325 / quill | +1.7700 | -1.4161 | -1.1871 | -1.1871 | -1.4161 | -0.9542 |
| Mercury_400828 / ember | +0.8577 | -1.1101 | -0.7788 | -0.7788 | -1.1101 | -0.7449 |
| Mercury_SC_400590 / ember | +1.8552 | +0.0672 | +0.2416 | +0.2416 | +0.0672 | +0.3191 |
| Mercury_405778 / ember | +0.8020 | -0.5773 | -0.3131 | -0.3131 | -0.5773 | -0.2469 |
| Mercury_SC_415473 / ember | -0.0436 | -0.5534 | -0.7344 | -0.7344 | -0.5534 | -0.6678 |
| Mercury_415083 / quill | +0.1041 | -0.2965 | -0.1000 | -0.1000 | -0.2965 | -0.1398 |
| Mercury_7164623 / quill | +0.1720 | -0.5936 | -0.3894 | -0.3894 | -0.5936 | -0.4349 |
| Mercury_7017973 / ember | +0.2844 | -1.3170 | -0.9552 | -0.9552 | -1.3170 | -1.0585 |
| Mercury_7058135 / quill | +0.8722 | -0.1398 | -0.2162 | -0.2162 | -0.1398 | -0.1279 |
| MEAP_2005_8_13 / quill | +0.9178 | -0.3396 | -0.4863 | -0.4863 | -0.3396 | -0.4812 |
| Mercury_175053 / ember | +0.2207 | -0.9193 | -0.8037 | -0.8037 | -0.9193 | -0.7106 |
| Mercury_7205800 / quill | +1.0889 | -0.3199 | -0.2423 | -0.2423 | -0.3199 | -0.3238 |
| ACTAAP_2010_7_12 / quill | +0.2281 | -0.7986 | -0.6257 | -0.6257 | -0.7986 | -0.7018 |
