# Windows 自动补录其他语言适配

日期：2026-10-01。上游仍为 `v1.3.3` / `f8427da`。本次基于 Windows 定制版 `05b1c5f`，交付版本 `1.3.3-windows.2`。

## 使用方式

选择 **WhisperX** 和多语言模型（例如 large-v3 或 large-v3-turbo），源语言选择韩语，再开启 **自动补录其他语言**。Windows CUDA 和 CPU 使用同一补录流程。选择“自动检测”时多语言检查自动启用；英语专用 `.en` 模型无法补录其他语言。

设置页和转录工作区均已解除 Apple Silicon 限制。识别出其他语言且缺少对齐模型时，继续使用原有下载后重试、保留句级时间戳或忽略该语言的交互。

## 实现与保留项

- Windows 使用当前 WhisperX 已加载的 FasterWhisper/CTranslate2 模型进行语言概率探测及局部复听，不额外加载第二份模型。
- 与 MLX 共用上游的 VAD 门限、持续语言证据、CJK 字符证据、歧义拒绝和区间合并逻辑；Windows 不导入 MLX。
- 复听使用原语言转录和真实词时间戳，裁去上下文词，保留韩语词间空格。外语结果质量低、主语言残余过短或无法可靠解码时，整个候选补录回退到原字幕。
- 外语独占结果使用其对应对齐模型；手工指定的韩语模型不再被用于英语等补录分组。
- 缓存区分真实模型、后端、补录开关及算法修订，避免复用旧的单语言结果。
- 保留之前的韩语断句与提示词、三份字幕同时保存、原文侧文件再翻译、Windows 保存和预览保护、CUDA/CPU 选择及隐藏工作进程等定制。

## 验证记录

- 完整离线 Python 回归：2974 passed、17 skipped、35 deselected。本次新增 27 项测试覆盖实际调用路径、保词、低质量解码、模型决策和取消。
- 既有 WhisperX 相关回归：83 项通过。
- 前端 Vitest：62 项通过；预览保护 Node 测试：6 项通过；ESLint 通过。
- Ruff 通过；完整 Pyright 检查 0 errors / 46 warnings；最终补录适配器单独检查 0 errors / 0 warnings。
- 用本机 Microsoft Heami 和 Zira 生成 47.944 秒韩语→英语→韩语合成音频，实际 `SubForge.exe` 完成 CUDA（large-v3-turbo / float16）和 CPU（tiny / int8）推理。两者均补出完整英语段落，保留各自初次识别的全部韩语条目，最终时间轴不交叠。
- 合成样本暴露的边缘丢词已修复并加回归：补录窗口可在额外语言证据支持下向空白区间扩展最多 2 秒；跨越边界且原文有支持的词保留，不引入完全处于区间外的上下文词。
- 实际桌面自检返回 0：FastAPI/HTTP、FFmpeg/FFprobe、whisper.cpp、DeepFilterNet3 推理、FasterWhisper/Silero VAD、WhisperX CUDA/forced alignment、pyannote 导入通过。
- 前端产物 27 个文件逐一哈希一致；最终补录模块重新编译进 PyInstaller 无归档模块目录，其代码对象与最终源码完全一致，并再次运行上述 CPU/CUDA 样本。
- 开发环境曾在 ONNX 原生库导入阶段长时间等待，实际推理验证采用清理重复 CUDA/OpenMP DLL 后的桌面包。
- 以上为合成音频功能验证，不代表真实韩语长视频的识别准确率、完整翻译流程或覆盖安装均已实测；未调用付费 API。

## 回退与产物

- 代码备份：`backup/windows-before-multilingual-20261001`。
- 旧桌面目录：`artifacts/SubForge-1.3.3-windows.1-backup-20261001`。
- 旧版 setup 安装包保留；本次没有覆盖已安装的程序，也没有推送或发布 Release。
- 构建、回归和音频样本证据保存在本地 `artifacts/multilingual-*`，不提交模型及生成音频。

## 构建方式

```powershell
$env:SUBFORGE_BUILD_VERSION='1.3.3-windows.2'
.venv/Scripts/python.exe scripts/build_desktop.py --no-archive
& work-dir/InnoSetup6/ISCC.exe /DMyAppVersion=1.3.3-windows.2 /DMyCompression=lzma2/normal scripts/windows_installer.iss
```

安装程序沿用原 AppId、中文界面和炉子图标，本地包使用 normal 压缩档。最终安装包大小及 SHA-256 以同目录验证文件为准。

## 最终安装包

交付完成：2026-10-02。源码提交：ce5debc。

- 路径：artifacts/SubForge-1.3.3-windows.2-windows-x64-setup.exe。
- 体积：2294343762 字节（2.29 GB）。
- ProductVersion：1.3.3-windows.2；Inno Setup 日志确认 Successful compile。
- SHA-256：7f394af3612cdf5b209617790eb2637ccb26a1564d701045886e66d79e8bc80e。
- 同目录提供 .exe.sha256；结构化验证记录：artifacts/multilingual-installer-verification.json。
- 未进行覆盖安装；旧版安装包及旧桌面目录均保留。
