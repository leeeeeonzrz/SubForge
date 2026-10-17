# Windows 降噪后音频读取修复

日期：2026-10-17。基于已同步上游 v1.3.7 的 Windows 定制分支。

## 故障与修复

用户的 3511.531 秒 MP4 在 DeepFilterNet3 完成降噪后，进入
`ChunkedASR._split_audio → load_audio_file → pydub.from_file` 时失败：
`ffmpeg returned error code: 3221225794`。

原实现向 pydub 传入已打开的文件句柄，没有提供格式。pydub 无法据此识别
WAV 扩展名，因此本来可直接读取的 PCM WAV 又经过 FFprobe/FFmpeg。
现在先用标准库 `wave` 确认 PCM，重置流位置，再明确以 WAV 格式读取。
文件句柄仍在所有路径上确定关闭；非 PCM、浮点 WAV、MP3 和未知扩展名
保持原来的自动检测路径，文件 I/O 错误不被吞掉。

不能仅凭 WAV 后缀启用 pydub 原生读取：它会误读 Extensible FLOAT 的采样
幅度。新增标准 FLOAT 和 Extensible FLOAT 的真实采样值回归保护这一点。

未修改 DLL、系统设置、识别模型或用户转录设置。短音频上的 DLL 搜索目录及
隐藏进程 A/B 探针均不能重现故障，所以没有将 DLL 冲突当成已证实原因。

## 原片与回归证据

- 在工作区冻结程序中对原片完整降噪；同一输出使用旧读取逻辑，复现完全相同的
  `3221225794`；改为 PCM 直接读取后成功读取全部 3511531 毫秒。
- 实际分为六块，前五块各 600 秒，末块 561.531 秒（含既有十秒重叠规则），
  所有 MP3 块重新解码成功。
- 从增强结果的第 30–54 秒取样，运行真实 WhisperX large-v3-turbo、CUDA
  float16 识别和韩语对齐，生成 12 个带时间轴的字幕片段。
- 这是全长降噪/分块及 24 秒识别验证，不代表整部视频已完成转录或翻译。
- 证据位于忽略目录 `artifacts/fix-ffmpeg-full-regression.json`，不提交原片、
  音频样本、模型或字幕内容。
- 新增与启动器相关的 30 项测试通过；Ruff 通过，定向 Pyright 0 errors。
- 全量离线回归 3110 项通过；另六项在仓库内自定义临时目录下因 Git 发现及
  路径授权测试前提改变而失败，恢复默认系统临时目录后六项全部通过。
  总计覆盖 3116 个通过项，17 skipped、35 integration deselected。

## 打包检查补全

旧打包检查只确认降噪文件生成，没有验证后续切块解码。新的
`SUBFORGE_CHECK_DENOISE` 检查调用真实 `_split_audio()`，再逐块解码 MP3；
按输入时长最多分三块，避免长素材产生大量诊断进程。失败详情写入诊断文件，
退出前清理临时增强文件；`smoke_desktop.py` 会显示这些错误。

桌面构建版本为 `1.3.7-windows.2`，源码版本为 `1.3.7+windows.2`。
前端以 webpack 生产构建完成；其余仍使用现有 Windows 打包流程。

- 修复提交：`c151731`。
- 桌面构建返回 0；冻结后的 `audio_io.pyc` 确认包含 PCM 检查和流复位。
- 36 个前端文件与生产导出逐一 SHA-256 一致；重复 CUDA/OpenMP DLL 已按
  既有流程去除。
- 新包完整 `smoke_desktop.py` 返回 0，包含 FastAPI/HTTP、FFmpeg/FFprobe、
  whisper.cpp、DeepFilterNet3 及 MP3 分块衔接、FasterWhisper/Silero ONNX、
  WhisperX/PyTorch CUDA/对齐和 pyannote 导入。
- 安装程序使用 `lzma2/fast` 压缩，不改变程序内容。

## 安装程序

- `artifacts/SubForge-1.3.7-windows.2-windows-x64-setup.exe`，3,057,846,188 bytes。
- SHA-256：`ffc2818c436825ff5404992c8376663b7b2031b468b5e3bad7c282c0ee46aa48`。
- 编译返回 0；上一版 `1.3.7-windows.1` 安装程序保留用于回退。

## 覆盖安装与安装后验收

- 安装到既有当前用户目录 `C:\Users\xuleh\AppData\Local\Programs\SubForge`，
  静默覆盖安装返回 0，安装日志确认成功且无需重启 Windows。
- 注册表及已安装后端均报告 `1.3.7-windows.2`；安装后的 EXE 和音频读取
  模块与已验证构建的 SHA-256 一致。
- 原视频和 settings.json 的 SHA-256 安装前后一致；81 个模型文件的相对
  路径和大小清单一致。
- 已安装程序通过真实上传及 `/api/transcribe/start` 路由处理原片第 30–54 秒
  的原始音频样本，使用现有配置完成降噪、切块、WhisperX CUDA float16、
  韩语对齐和 SRT 输出。任务 `completed`、进度 100、error 为 null，产生
  11 个带时间轴的字幕片段。该样本重新独立降噪，与前述全片降噪后切出的
  样本略有不同。
- 验证证据：`artifacts/fix-ffmpeg-installed-verification.json`、
  `artifacts/fix-ffmpeg-installed-transcription.json` 和安装日志。
- 未调用付费翻译 API，未推送或发布；新程序已启动供用户继续使用。
