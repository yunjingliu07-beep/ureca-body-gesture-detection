# 配置

从仓库根目录执行：

```sh
cp configs/classroom.example.json configs/classroom.local.json
python -m classroom_behaviour --config configs/classroom.local.json --show-scene
```

根据实际机位修改区域坐标和朝向。`*.local.json` 不提交到 Git。
配置字段与行为规则见 [behaviour_rules.md](../docs/behaviour_rules.md)。
