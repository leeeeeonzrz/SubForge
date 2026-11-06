# Windows 定制版同步上游 v1.3.9

日期：2026-11-06。桌面/安装器版本：`1.3.9-windows.1`；源码版本：`1.3.9+windows.1`。

## 基线与恢复

- 上游：`470e4329c994270fe94af01326a3c5489e474c77`（`v1.3.9`），包含 v1.3.8 和 v1.3.9。
- 原有源码：`6524e4f`。上次尚未提交的韩语会话优化完整保存为 `9d8a5f8`。
- 本次使用保留双方历史的 merge，合并提交为 `61abcae126609b6427b6c4960ea136723a620260`。
- 恢复分支：`codex/backup-before-upstream-20261106` 和 `codex/backup-with-korean-before-v1.3.9-20261106`。
- 本地 `master` 快进至上游提交；开发继续使用 `codex/windows-main`。
- 旧工作区程序目录保留在 `artifacts/SubForge-before-v1.3.9-20261106`；旧安装包保留。

## 合入和保留

合入上游字幕结构规划、车型和规格完整性、阅读预算、联网术语核实、恢复区域复核、未保存编辑保护和本地 HTTP/WebSocket 来源保护。

保留 Windows CUDA/CPU 与隐藏工作进程、FasterWhisper/WhisperX 模型共享、多语言自动补录与按语种对齐、韩语短答/否定/强调/拟声保护、原文/译文/双语三份字幕、原文侧文件再翻译、预览修订保护、PCM WAV 降噪衔接修复、炉子图标及中文安装器。上游删除的免费模型面板/API/扫描服务、硬件状态及模型推荐控件继续保留。

两套同名重复复核模块按职责合并：上游通用复核仍为 `repetition_review.py`，本地韩语证据规则迁到 `korean_repetition_review.py`。真实缺口优先，随后韩语严格复核，再由通用复核使用剩余预算；总计最多三个目标窗口、180 秒解码。通用规则不能绕过韩语短答、否定和独立对齐保护，重复警告不能掩盖真实时间轴缺口。

后端及桌面自检按三份输出与原文指针约定适配，保留来源校验和精确文件授权。上游新增测试的 UTF-8 夹具读取显式指定编码，使中文 Windows 默认 GBK 环境也可验证。

## 源码验证

- 完整离线 Python：3545 passed、17 skipped、35 external integration deselected（73.00 秒）。15 个跳过项是验证虚拟环境未启用原生 VAD，另两个是仅限 macOS 的 DMG 测试。
- 前端 Vitest：77 passed；独立预览保护：6 passed；ESLint 通过。
- 前端 webpack、TypeScript、静态导出通过，版本 `1.3.9-windows.1`，36 个输出文件。
- Ruff 通过；Pyright：0 errors、13 warnings（可选 Apple MLX 依赖）。
- Windows 沙箱曾阻止原子替换和 asyncio 本地 socketpair；失败日志保留，正式完整回归在允许本机 IPC/文件替换的环境运行。
- 依赖声明与锁文件未变化，无训练、模型替换或新增模型下载。

验证证据保存在忽略目录 `artifacts/sync-v1.3.9-*`，源码提交和 Git 树记录在 `sync-v1.3.9-source-verification.json`。

## 新程序包验证

- 使用已有 `.venv`、缓存 FFmpeg/whisper.cpp 和上述前端导出完成构建；`scripts/build_desktop.py --no-archive --skip-frontend-build` 返回 0。
- 冻结版本正确，36 个前端文件 SHA-256 匹配；15 个关键模块的冻结字节码与已验证源码逐一匹配，包含两套重复复核、翻译、原文保护及恢复的免费模型服务。
- 程序目录共 20,684 个文件、5,476,057,509 bytes，较上版已验证分卷包的程序目录增加 192,369 bytes（约 188 KiB）。CTranslate2 重复 CUDA/OpenMP DLL 已移除，保留 Torch 共享运行库。
- 新 EXE 的 FastAPI/HTTP 来源保护、三文件保存、FFmpeg/FFprobe、whisper.cpp、DeepFilterNet3 实际降噪与 MP3 切块衔接、FasterWhisper/CTranslate2/PyAV、CUDA 张量及 pyannote 导入检查全部通过。基础 smoke 中的 alignment 检查仅为导入，不将其计为实际词级对齐。
- 47.944 秒韩语→英语→韩语合成样本：CUDA large-v3-turbo 和 CPU tiny 均保留全部七条原有韩语字幕，完整补出预期英语段落，无时间轴重叠；耗时分别为 25.688 秒和 21.844 秒。
- 15 秒现有韩语拟声对照完成真实 CUDA 转录与韩语词级强制对齐：七个词段时间戳有效，六次正常“삐용”全部保留，文本与此前对照一致，无覆盖问题。该检查临时插桩工作区程序并隔离数据目录；33 个插桩模块及整个 20,684 文件程序目录在结束后全部恢复并逐文件 SHA-256 匹配，原音不变、工作进程数归零。它验证短片段的插桩冻结运行环境，不是整场直播准确率评估。

## 安装程序

目录：`artifacts/release-v1.3.9-windows.1/`。运行 EXE 时两个 BIN 必须保持原文件名并位于同一目录。

| 文件 | Bytes | SHA-256 |
| --- | ---: | --- |
| `SubForge-1.3.9-windows.1-windows-x64-split-setup.exe` | 4,178,879 | `04126f9b090ce2bcfd102c5962653e1ad9896d69c5a8adfcfb623f50bfc4ffe8` |
| `SubForge-1.3.9-windows.1-windows-x64-split-setup-1.bin` | 1,895,821,056 | `e1f9b1d875b0f5193d640c17bd3f9d6f00e32745daf7a7e77ed9d3bd4fb8006b` |
| `SubForge-1.3.9-windows.1-windows-x64-split-setup-2.bin` | 1,337,999,966 | `2ce473e70ad9fcc7f1a0a6decdaa16e196680622bdbf794e1432e85713f15ca2` |

Inno Setup 6.7.3 编译成功，耗时 421.843 秒；安装器产品版本正确。使用与上一版相同的 ZIP 压缩与分卷参数，每个文件小于 2 GiB。目录内另有 `SHA256SUMS.txt` 和 `安装说明.txt`。

使用 innounp 2.67.11 直接解包，返回 0、耗时 163.485 秒。安装载荷的 20,684 个文件、5,476,057,509 bytes 与输入清单完全一致，逐文件 SHA-256 全部通过。记录：`artifacts/release-v1.3.9-windows.1-verification.json`；本轮仅构建和验证，没有执行覆盖安装或发布 GitHub Release。

## 验收边界

Windows 默认未包含 TEN-VAD 原生库。上游通用循环自动替换要求 TEN 与 Silero 的独立证据，因此缺少 TEN 时保留原文并提示复核；不宣称该路径已在 Windows 自动修复所有循环。原有韩语 Silero 与独立词对齐路径保留。

离线回归和短音频运行检查不代表整场直播准确率保证；没有付费在线翻译或整片人工语义验收。
