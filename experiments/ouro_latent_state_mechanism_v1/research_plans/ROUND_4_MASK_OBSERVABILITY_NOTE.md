# ROUND_4 mask and backend observability note

实际传到attention/SDPA的mask参数允许为None；实际position/RoPE/cache与基于pinned adapter的因果规则分别验证。未观测底层kernel身份保持unknown。配对同题相同shape/状态的实际参数一致性是本实验门槛；如使用透明API observer，需独立no-observer/full-logits零差锚，禁止为观察更换SDPA backend。与ABCA不同长度prefix逐位问题不混用未通过组件。
