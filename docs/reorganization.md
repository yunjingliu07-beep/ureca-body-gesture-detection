# 仓库整理记录

整理日期：2026-11-13。

代码、配置、文档、输入视频和分析结果已按用途归类。清理和去重共减少
1,261,340,073 字节，约 **1.26 GB（1.17 GiB）**。原始视频、模型权重、
实验数据、验证报告、第三方源码和可用运行环境均保留。

## 新目录

```text
.
├── README.md                    项目说明与运行命令
├── requirements.txt             基础 Python 依赖
├── requirements/objects.txt      教具检测的可选依赖
├── classroom_behaviour/          分析代码与 HMAD R 桥接
├── scripts/                     本地依赖构建、模型下载
├── configs/                     示例配置及本地配置
├── data/videos/                 原始讲课视频
├── outputs/
│   ├── runs/                    两段讲课视频的正式结果及后续实验
│   ├── validation/              已核验的分析与模型测试
│   └── archive/                 历史、调试及未完成结果
├── logs/                        构建、运行和环境迁移记录
├── docs/                        行为规则、第三方说明和整理记录
├── tests/                       56 项回归测试
├── models/                      模型与下载说明
├── external/                    固定版本的第三方源码及构建产物
└── .runtime/                    本地 OpenFace/R 原生运行环境
```

`external/` 的构建目录和 `.runtime/` 含编译时的绝对路径，保留原位置以便继续使用。
`.vscode/` 的解释器设置继续指向现有 `ureca` Conda 环境。

## 主要名称变更

| 原路径 | 新路径 |
| --- | --- |
| `main.py` | `classroom_behaviour/cli.py`；入口为 `python -m classroom_behaviour` |
| `behaviour_common.py` | `classroom_behaviour/temporal.py` |
| `tracking_config.py` | `classroom_behaviour/config.py` |
| `face_hand_landmarks.py` | `classroom_behaviour/landmarks.py` |
| 其他识别模块及 `hmad_bridge.R` | 保留描述性名称，移至 `classroom_behaviour/` |
| `build_native.py` | `scripts/build_native.py` |
| `setup_tracking.py` | `scripts/setup_models.py` |
| `classroom_config.example.json` | `configs/classroom.example.json` |
| `BEHAVIOUR_RULES.md` | `docs/behaviour_rules.md` |
| `THIRD_PARTY.md` | `docs/third_party.md` |
| `requirements-objects.txt` | `requirements/objects.txt` |
| `tests/test_face_hand_landmarks.py` | `tests/test_landmarks.py` |
| 根目录的两个长文件名 WebM | `data/videos/calculus_lecture.webm`、`macroeconomics_lecture.webm` |
| `output/` 中的正式结果 | `outputs/runs/` |
| `outputs/` 中的验证及历史结果 | 分别归入 `outputs/validation/` 和 `outputs/archive/` |
| `.runtime/run-logs/` | `logs/runs/` |
| `.runtime/python-env-migration/` | `logs/environment_migration/` |

结果文件统一采用 `calculus_lecture_*`、`macroeconomics_lecture_*` 前缀；
OpenFace/HMAD 内部文件名保留上游约定。所有移动、改名、删除与去重详情见
[reorganization_manifest.json](reorganization_manifest.json)。

## 清理内容

| 类型 | 处理及原因 |
| --- | --- |
| Conda 下载包 | 删除 175 个 `.conda` / `.tar.bz2` 安装归档；已安装环境和解包目录保留 |
| Python、Matplotlib 和 Finder 缓存 | 删除可重建的字节码、字体缓存和 `.DS_Store` |
| 临时运行状态与 CLI 帮助副本 | 运行已通过，验证报告保留；帮助可随时通过 `--help` 查看 |
| 旧 BAGMAD 调试脚本 | 已由正式 R 桥接取代；相关诊断输出归档保留 |
| HMAD 工作副本 | 7 个视频/CSV 文件经 SHA-256 核对相同后，改为指向本次实验原文件的相对符号链接，节省约 862 MB |

归档结果包含旧版本与未完成运行。它们用于复现实验或诊断历史问题，不能用于声称
当前模型已通过验证。每次复制实验请保留整个实验目录及符号链接。

## 引用与历史记录

代码导入、默认输入和输出、配置路径、依赖文件引用及文档命令已更新。
结构化元数据、结果清单与 ELAN 的绝对/相对媒体链接已同步。
历史日志原文保留，其中的旧路径可通过整理清单对照。

有两类原本已失效的历史来源记录继续保留，避免改写实验历史：

- 两分钟验证结果中的三处 `openface_input` 指向早已不存在的失败运行目录。
  可用的 CSV 副本保存在该实验的 `native/openface.csv`，其内容与记录的哈希一致。
- 环境迁移清单中的 `source` 指向已迁走的 `.venv`，当前使用 `ureca` Conda 环境。

旧 `openface_hmad_30s_v2` 归档的 ELAN 链接原本指向缺失的 `analysis.avi`。
核实其为完整画面、从零开始的同一时间轴后，链接已改为保留的原始视频。

## 验证

- 整理前后均有 56 项自动测试通过；测试中的导入与默认路径断言已同步更新。
- 主程序、统计工具及模型下载工具的模块入口可用；依赖检查通过。
- 两段正式结果重新生成的窗口 CSV 与已有文件逐字节一致，均包含 4 个时间窗口。
- 本地原生 CSV 的来源哈希保持匹配，7 个相对链接均可访问。
- 使用真实视频与已有原生结果，成功完成 0.2 秒、6 帧的 MediaPipe 身体/手部分析、
  视频导出与窗口统计。此检查需要 macOS 图形服务，已在可访问该服务的执行环境完成。
- 使用迁移后的 R 桥接文件，对已有 30 秒 OpenFace 测量重新运行 HMAD；全部组件成功。

验证只检查整理后的可运行性与数据完整性；模型准确率仍按行为规则文档中的研究条件评估。

## 常用命令

在仓库根目录执行：

```sh
conda activate ureca
python -m classroom_behaviour --check-dependencies
python -m classroom_behaviour --output-dir outputs/runs/new_experiment
python -m classroom_behaviour.window_summary \
  --metadata outputs/runs/calculus_lecture/calculus_lecture_metadata.json
python -m scripts.setup_models --help
python -m unittest discover -s tests -v
```

需要重新构建原生依赖时运行 `python -m scripts.build_native`。仅运行已有项目无需重建。
