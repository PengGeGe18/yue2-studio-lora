# yue2-lora

从 [YuE2-Studio](https://github.com/timoncool/YuE2-Studio) 中抽取出的 **YuE2 LoRA / LoKr 训练逻辑**，做成一个没有 UI 的纯命令行项目：写一个配置文件、跑一条命令，自动下载模型并完成训练。

- 无 Web 界面、无服务端，只有 CLI 和配置文件
- 训练配方（fast / balanced / thorough / tuned）、数据准备阶段、断点续训与 YuE2-Studio 完全一致
- 内置模型自动下载：Hugging Face 固定 revision + SHA-256 校验 + 断点续传
- 多平台：Windows 自动获取预编译训练器；Linux 自动从固定 commit 构建；macOS 可运行本工具，但联合训练内核仅支持 CUDA
- 纯 Python 标准库实现，无第三方依赖（Python ≥ 3.8）

> 说明：YuE2-Studio 里的“LoRA 训练逻辑”分为两层。真正的训练内核是
> [HOT-Step-CPP](https://github.com/timoncool/HOT-Step-CPP) 的 `ace-train`（C++/ggml，CUDA）；
> YuE2-Studio 自己负责的是数据准备、配方、参数拼装、模型下载和 checkpoint 管理。
> 本项目抽取的是后者——它决定了训练怎么跑、跑多少步、用什么参数——并把训练器编排成一条命令。

## 安装

```bash
git clone <this repo> yue2-lora && cd yue2-lora
python3 -m pip install -e .
```

## 快速开始（一条命令训练）

```bash
# 1) 生成配置和目录骨架
yue2-lora init

# 2) 把音频（`.wav`/`.flac`/`.mp3`/`.ogg`/`.m4a`）放进 dataset/audio，并为每首准备 <stem>.lrc 歌词
#    <stem>.yue2.txt 可省略——缺失时会用听辨模型（MOSS-Music，"按声音描述"）自动生成
#    （若想手动指定风格，直接放 <stem>.yue2.txt，一行纯文本即可）

# 2.5) 首次使用，先安装听辨模型（约 10 GB；没有它就无法自动生成风格）
yue2-lora caption -c config.toml --install

# 3) 生成标签文件（以及可选的 48 kHz 音频）
yue2-lora dataset -c config.toml

# 4) 训练：自动下载缺失模型并开始训练
yue2-lora train -c config.toml
```

只想先看看会执行哪些命令，不实际训练：

```bash
yue2-lora stages -c config.toml        # 打印全部阶段命令
yue2-lora train -c config.toml --dry-run
```

## 数据集格式

```
dataset/audio/
  my-song.wav          # 48 kHz WAV
  my-song.lrc          # 歌词（LRC，时间戳会自动去掉）
  my-song.yue2.txt     # 一行风格（可省略；缺失时由听辨模型生成）
```

每首歌的标签文件 `my-song.txt` 由 `yue2-lora dataset` 生成，你只需要提供音频和 `.lrc`；
风格文件 `my-song.yue2.txt` 若不存在，会用听辨模型生成。
`train` 只检查这些文件是否齐全，不会自动生成；缺文件时会退出并提示先运行 `yue2-lora dataset`。

1. **风格（按声音描述）**：优先读 `my-song.yue2.txt`（只有一行）。
   若它和已有的 `my-song.txt` caption 都不存在，`dataset` 会调用 **MOSS-Music** 听这首歌，
   与 YuE2-Studio 点击「按声音描述」的逻辑一致：
   - `ace-caption` 让 MOSS 听音频，按固定 5 段模板输出 `genre / caption / bpm / key / signature`；
   - 风格句由 `"<genre>. <caption>"` 组成，末尾补上 `bpm`，即 `…, 72 BPM`（与 Studio 的
     `normalize_yue2` 完全一致）。
   想手动覆盖，直接在 `.yue2.txt` 里写一行纯文本即可。
   > 说明：Studio 会用 Beat This!（ONNX）修正 MOSS 猜的 BPM；纯标准库 CLI 无法跑 ONNX，
   > 这里直接用 MOSS 在 `bpm:` 字段给出的值。Studio 的 AI 写作助手（llama.cpp）是可选增强，
   > 没有它时 Studio 同样直接采用 MOSS 听到的描述——这正是本项目的实现。
   >
   > **安装提示**：`--install` 直接下载官方预编译包（`scragnog/HOT-Step-CPP` release `v1.6.0`，
   > timoncool README 的 “Download the latest release” 就指向它），解压出与 libggml 同目录的
   > `engine/ace-caption`，**代码里不做源码编译**。模型权重始终来自 HuggingFace（Q8_0，流式下载）。
   > 平台默认：macOS Apple Silicon → Metal；Linux/Windows x64 → CUDA 12.8。
   > 其他后端用环境变量 `YUE2_CAPTION_VARIANT` 选择，例如 `linux-x64-cuda13.1`、
   > `linux-x64-vulkan`、`linux-x64-rocm`、`win-x64-vulkan`、`*-cpu`。
   > Intel macOS 没有官方预编译，需手动构建一次，并用 `[dataset] captioner` 指向它。
2. **歌词**：从 `my-song.lrc` 提取，自动去掉 `[mm:ss.xx]` 时间戳和 `[ti:]`、`[ar:]` 等元数据行；
   `[Verse]`、`[Chorus]` 这类段落标记会保留。
3. **合并**：先生成/读取 `my-song.yue2.txt`，再写出 `my-song.txt`：

```
caption: <my-song.yue2.txt 的内容>
is_instrumental: false
lyrics:
<从 my-song.lrc 提取的歌词>
```

`is_instrumental` 默认为 `false`。纯音乐请在一个已存在的 `my-song.txt` 里写 `is_instrumental: true`，
这时不需要 `.lrc`。

`my-song.yue2.txt` 是给 planner 的专用一句话风格，`my-song.txt` 的 `caption:` 与它保持一致；
两者的区别只是 `.txt` 还要承载歌词和器乐标记。

采样率不是 48 kHz 时，把 `[dataset] normalize = true` 打开，会用 ffmpeg 转成 48 kHz 16-bit WAV 后再训练（需要 ffmpeg 在 PATH）。

## 配置说明

完整示例见 [config.example.toml](config.example.toml)。

| 段 | 键 | 说明 |
| --- | --- | --- |
| `workspace` | `dir` | 工作目录，模型、运行记录、训练器都在其下 |
| | `models` / `runs` / `trainer` | 各子目录名 |
| `dataset` | `audio` | 音频文件夹 |
| | `trigger` | 适配器唤醒词，会加进风格前缀 |
| | `caption_seconds` | 听辨模型每首最多听多少秒（默认 120，超长取中段） |
| | `captioner` | 指定已有 `ace-caption` 路径（默认自动查找/下载官方预编译） |
| | `vocals` | 歌词时间对齐用的 `<song>/vocals.wav` 目录（仅 `tuned` + `lyric_timing` 需要） |
| | `normalize` | 是否用 ffmpeg 统一转 48 kHz |
| | `prepared` | `normalize = true` 时 48 kHz 音频的输出目录 |
| `model` | `base` / `companion` / `tokenizer` | 留空则自动下载到 `models/`；填路径则使用已有文件 |
| `train` | `device` | `CUDA0` / `Vulkan0` / `MTL0` / `CPU` |
| | `ffmpeg` | ffmpeg 可执行文件，`none` 表示关闭 |
| | `trainer` | 指定已有 `ace-train`/`music-train` 路径 |
| `recipe` | `preset` | `fast`(100 步) / `balanced`(200 步) / `thorough`(300 步) / `tuned`(旧配方) |
| | `seed` | 随机种子 |
| | 其余 | 仅 `tuned` 生效：`stop`、`epochs`、`steps`、`save_every`、`target_kl`、`adapter`(`lokr`/`lora`)、`rank`、`alpha`、`lokr_dim`、`lokr_factor`、`optimizer`(`prodigy`/`adamw`)、`learning_rate`、`planner_lr_scale`、`lyric_timing`、`cursor_weight` |

预设配方固定为：`lokr` dim 128 / factor 4 / alpha 256，AdamW(`adamw-lm`) 1e-4，cosine-floor，60 秒解码窗口，关闭 KL 停止与歌词时间对齐。这与 YuE2-Studio 的 base-matched 配方一致。

## 命令

| 命令 | 作用 |
| --- | --- |
| `yue2-lora init` | 生成 `config.toml` 与目录骨架 |
| `yue2-lora download` | 下载并校验全部固定版本模型（含 companion、tokenizer），Windows 上还包括训练器；`--with-caption` 同时拉取 MOSS-Music |
| `yue2-lora doctor` | 检查数据集、模型、训练器、听辨模型、ffmpeg、设备 |
| `yue2-lora caption --install` | 下载官方预编译 `ace-caption`（HOT-Step v1.6.0）与 MOSS-Music（~10 GB） |
| `yue2-lora caption` | 查看听辨器与 MOSS 模型是否就绪 |
| `yue2-lora dataset` | 为缺失风格生成 `.yue2.txt`（按声音描述）、写出 `.txt`，可选转 48 kHz，并校验数据集 |
| `yue2-lora stages` | 打印 `yue2-preprocess` / `-tokenize` / `-align` / `-sheet` / `-prepare-aitk` / `-joint-train` 命令 |
| `yue2-lora train` | 一键训练：检查数据集 → 补模型 → 跑完所有阶段 |
| `yue2-lora train --resume` | 从最近一次运行的 checkpoint 继续训练 |
| `yue2-lora checkpoints` | 列出运行产出的 checkpoint 与可续训点 |
| `yue2-lora trainer --install/--build` | 单独安装（Windows 预编译）或构建（Linux 源码）训练器 |
| `yue2-lora info` | 打印合并后的有效配置 |

## 训练流程

`train` 会依次执行（与 YuE2-Studio 的 `training_stages` 一一对应）：

1. **latents** `yue2-preprocess` — 音频 → VAE latent + `yue2_preprocess.json`
2. **codes** `yue2-tokenize` — 语义 tokenizer → `codec_ids`
3. **align** `yue2-align` — 仅 `tuned` + `lyric_timing`：MMS 强制对齐歌词
4. **scores** `yue2-sheet` — SheetSage2 生成 ABC 乐谱
5. **prepare** `yue2-prepare-aitk` — 组装 `prepared/dataset.json`
6. **train** `yue2-joint-train` — AR+NAR 联合训练，写出 `output/checkpoint-step<N>/`

产物在 `runs/<时间>-<配置名>/`：

```
runs/20261126-101500-config/
  run.json
  cache/yue2_preprocess.json
  prepared/dataset.json
  output/checkpoint-step100/…safetensors
```

## 断点续训

训练中断或想多训一些：

```bash
yue2-lora checkpoints -c config.toml     # 查看可续训的 step
yue2-lora train --resume -c config.toml  # 从最新 optimizer.resume 继续，写入 output-2/
```

续训沿用 `run.json` 里保存的配方；`optimizer.resume` 会做 44 字节头与长度校验，不完整的记录会被跳过。

## 跨平台说明

| 平台 | 训练器来源 | 说明 |
| --- | --- | --- |
| Windows x64 | 自动下载 YuE2-Studio 预编译 `hotstep-tools-…-cuda-windows-x64.zip`（`music-train.exe`） | 需 NVIDIA 显卡与 CUDA 运行库 |
| Linux | `yue2-lora trainer --build` 从固定 commit 克隆 HOT-Step-CPP 并用 CMake + CUDA 构建 `ace-train` | 需要 `git`、`cmake`、CUDA Toolkit |
| macOS | 本工具可正常运行，但联合训练内核依赖 CUDA，无法本地构建 | 在 CUDA 机器上构建后，用 `[train] trainer` 指向该二进制 |

### 为什么听辨器不编译、训练器仍可能需要编译

- **听辨器 `ace-caption`：改为不编译。** 官方 `v1.6.0` 预编译包覆盖 macOS(ARM) / Linux x64 /
  Windows x64 的各后端，包内含 `engine/ace-caption` 及其 `libggml*` 依赖；实测其 `--help`
  与本项目调用参数完全一致，因此直接下载解压使用。
- **训练器 `ace-train`：确实需要编译，原因如下。**
  1. 本项目钉的是 `timoncool/HOT-Step-CPP@479cbeb8`，而 YuE2-Studio 只提供该 commit 的
     **Windows** 预编译包，Linux 没有对应产物。
  2. 官方 `v1.6.0` 的 `ace-train` 已改为子命令式（`preprocess` / `train-lm` / `train-dit` /
     `mm3-*` 等），与本项目固定 commit 使用的命令行**不兼容**，不能直接替换。
  3. YuE2 的联合训练损失仅 CUDA 可用，Linux 上还需按本机 GPU 架构编译。
  若希望完全避免编译，可在 CUDA 机器上构建一次，再用 `[train] trainer` 指向该二进制
  （Windows 已自动使用预编译 `music-train.exe`）。

模型下载、配置解析、数据准备、命令编排、checkpoint 管理在所有平台一致。

硬件要求：按默认配方约 **11 GB 显存**；模型磁盘占用约 **9.7 GB**（训练包约 8.3 GB + companion 134 MB + tokenizer 2.4 GB）。

## 与 YuE2-Studio 的对应关系

| YuE2-Studio | 本项目 |
| --- | --- |
| `crates/music-engine/src/yue_train.rs`（Recipe、presets、train_args、training_stages、sidecars、resume_point、checkpoints） | `yue2_lora/recipe.py`、`stages.py`、`checkpoints.py` |
| `crates/music-server/src/training.rs`（Dataset、sidecars、run 目录） | `yue2_lora/dataset.py` |
| `crates/music-server/src/listen.rs`、`prepare.rs`（MOSS 听辨、`parse_prose`、`normalize_yue2`、`heard_style`，"按声音描述"） | `yue2_lora/caption.py` |
| `crates/music-server/src/downloads.rs`、`model_manager.rs`（资产清单、SHA、下载） | `yue2_lora/assets.py`、`download.py` |
| `engines/music-train-source.json`（训练器来源） | `yue2_lora/trainer.py` |
| HTTP/UI 层 | 移除，改为 CLI + 配置 |

## 测试

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖配方预设与参数拼装、阶段顺序、sidecar 格式、TOML/JSON 配置解析、数据集读取、checkpoint 续训扫描与设备名归一化。

## 许可与致谢

- 本项目代码：MIT
- 训练器 [HOT-Step-CPP](https://github.com/timoncool/HOT-Step-CPP)：MIT
- YuE2 训练权重、decoder companion：**CC BY-NC 4.0（禁止商用）**，训练出的适配器继承该限制
- 逻辑来源：[YuE2-Studio](https://github.com/timoncool/YuE2-Studio)（MIT）
