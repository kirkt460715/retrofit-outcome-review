# 资料说明

各集合保存在 `data/`，JSON 对象的 `fields` 为字段列表，`key` 为集合主键，`rows` 为记录数组。各记录值均为字符串，空串含义以各集合的说明为准。

## buildings

楼栋、街区、共用通行系统及立项前需求组。

主键为 `building`，字段为 `building`、`area`、`way`、`baseline_group`。

## households

住户关系和相关观察分层。

主键为 `person`，字段为 `person`、`building`、`floor`、`group`。

## phases

实际开放或封闭的设施覆盖及计划起点。楼层边界包含端点。

主键为 `id`，字段为 `id`、`project`、`way`、`kind`、`first_floor`、`last_floor`、`planned`、`actual`、`close`、`state`。

## connections

住户可通往共用设施的有效旁路。

主键为 `id`，字段为 `id`、`building`、`way`、`valid_from`、`valid_until`、`first_floor`、`last_floor`。

## other_projects

同时开启且可能改变出行机会的项目，登记其范围和时间；登记不提供因果结论。

主键为 `id`，字段为 `id`、`area`、`scope`、`start`、`end`。

## observations

周记录，空次数不是零。

主键为 `id`，字段为 `id`、`person`、`week`、`state`、`outings`。

## proposals

待选的二期组合、试行承诺、共线路的设计。状态 issued 表示已签试建。

主键为 `id`，字段为 `id`、`state`、`building`、`way`、`path`、`kind`、`first_floor`、`last_floor`、`planned`。
