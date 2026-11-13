# 分析结果

| 目录 | 用途 |
| --- | --- |
| `runs/calculus_lecture/` | 微积分视频的 120 秒分析结果 |
| `runs/macroeconomics_lecture/` | 宏观经济学视频的 120 秒分析结果 |
| `validation/` | 自动或人工核验过的运行、原生模型和对象检测验证记录 |
| `archive/` | 历史版本、调试和未完成运行；用于追溯，不能当作最新验证结果 |

每次新分析使用 `runs/` 下的新目录，例如 `--output-dir outputs/runs/my_experiment`。
视频、逐帧 CSV、事件 CSV、窗口 CSV 和元数据采用统一的 `calculus_lecture_*`
或 `macroeconomics_lecture_*` 前缀。OpenFace/HMAD 的内部文件名保留其约定。

`native/hmad_workspace/` 中与本次运行的分析视频和 OpenFace CSV 完全相同的副本，
已改为指向同一 `native/` 目录内原文件的相对符号链接。复制单次实验时请复制整个实验目录，
并保留符号链接。各实验目录可独立保存；链接不跨实验目录。

元数据与 ELAN 媒体链接已同步到整理后的路径。历史日志保留原文，其中旧路径可通过
[整理记录](../docs/reorganization_manifest.json) 对照。结果文件不提交到 Git。
