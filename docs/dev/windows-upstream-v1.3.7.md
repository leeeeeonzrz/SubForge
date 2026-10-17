# Windows 定制分支同步上游 v1.3.7

日期：2026-10-17。

## 基线与回退

- 上游：`origin/master`，`25221ca`（`v1.3.7`）。
- 同步前 Windows 分支：`80bde04`；回退分支：`backup/windows-before-v1.3.7-20261017`。
- 在 `codex/windows-main` 执行保留合并拓扑的 rebase，将上次集成的 `f8427da` 更新为 `25221ca`。旧定制提交及其历史合并关系保留。
- Windows 源码版本：`1.3.7+windows.1`；桌面构建版本：`1.3.7-windows.1`。

## 合入与保留

本次合入上游 v1.3.4–v1.3.7 的转录最终覆盖检查、短语音缺口恢复、疑似非语音字幕提示、翻译衔接、中文范围副词边界及原生导入窗口修复。

保留 Windows/CUDA/CPU 与隐藏工作进程、FasterWhisper/WhisperX 模型共享、原文/译文/双语三份字幕、原文侧文件再翻译、文档保存和预览保护、韩语词组与断句/翻译规则、多语言自动补录及按语种对齐、炉子图标和中文安装程序。

对比 `f8427da..80bde04` 与 `25221ca..11d545f`：两者均包含同一组 62 个定制文件，逐文件增删行数相同。完整补丁忽略 Git 哈希与差异行号后，唯一差异是版本号。47 个文件的内容与同步前相同，其余文件叠加上游变更或更新版本号。

上游新增的转录→翻译交接测试原先从 `subtitle_file` 读取双语内容；本分支该字段始终指向原文侧文件。测试已适配为验证原文指针不变、三份文件路径独立且均获得访问授权，并分别检查原文、译文和双语内容；未修改此业务约定。

## 验证方法与环境

- Python 完整离线回归：3098 passed、17 skipped、35 deselected，耗时 66.35 秒。使用已有 `artifacts/venv-validation`，排除 `integration`；测试进程的 PATH 显式加入备份桌面包中的 FFmpeg，不改变系统设置，也避免与正在重建的运行目录竞争。
- 构建用 `.venv` 的 ONNX 原生模块在本机发生访问冲突，因此不以该环境的失败作为业务回归结论；桌面运行时另由新打包产物验证。
- 前端 Vitest：68 项通过；独立预览保护 Node 测试：6 项通过；ESLint 通过。
- 前端生产构建使用 `NEXT_PUBLIC_APP_VERSION=1.3.7-windows.1 npm run build -- --webpack`，TypeScript 和静态导出通过。默认 Turbopack 在本机发生 PostCSS 子进程握手失败；仅使用构建参数切换，未更改源码或默认配置。
- Ruff 通过。Pyright：0 errors、48 warnings（包含可选平台依赖等告警）。
- 测试、构建和运行证据保存于本地 `artifacts/rebase-v1.3.7-*`，不提交模型、音频和二进制产物。

## 新桌面包验证

- 使用 `.venv/Scripts/python.exe scripts/build_desktop.py --no-archive --skip-frontend-build`，构建版本环境变量为 `SUBFORGE_BUILD_VERSION=1.3.7-windows.1`。前端复用上文已完成的 webpack 生产构建。
- 构建返回 0；新包冻结后端版本正确，36 个前端文件逐一 SHA-256 匹配。CTranslate2 下的重复 `cudnn64_9.dll` 和 `libiomp5md.dll` 已移除，使用 Torch 配套共享运行时。
- 新包 `scripts/smoke_desktop.py dist/SubForge` 返回 0：FastAPI/HTTP、FFmpeg/FFprobe、whisper.cpp、DeepFilterNet3 实际推理、FasterWhisper/CTranslate2/PyAV、WhisperX CUDA/forced alignment 运行时、pyannote 导入均通过。
- 使用此前保留的 47.944 秒韩语→英语→韩语合成音频，对新 `SubForge.exe` 实测。CUDA（large-v3-turbo）保留 7 条韩语原句，CPU（tiny）保留 8 条韩语原句；两者均补出完整英语内容，最终时间段无重叠。对应结果和断言汇总保存在 `artifacts/rebase-v1.3.7-multilingual-verification.json`。
- 以上为合成音频及运行时验证，不代表真实长视频识别准确率、完整在线翻译、可见 UI 操作或覆盖安装均已实测；没有调用付费 API。
- 旧桌面目录已备份到 `artifacts/SubForge-1.3.3-windows.2-backup-20261017`，旧安装程序保留。未覆盖安装、推送或发布 Release。

## 安装程序

- 文件：`artifacts/SubForge-1.3.7-windows.1-windows-x64-setup.exe`。
- 体积：2,294,427,339 bytes；产品版本：`1.3.7-windows.1`。
- SHA-256：`e30221525e65dd7c429e0a6d59e8df73cd714ff796abc344b52acdf8777fdaf2`；同目录提供 `.exe.sha256`。
- Inno Setup 使用 `lzma2/normal` 编译成功，20,673 个源文件与已验证的新桌面包逐一匹配。
- 验证记录：`artifacts/rebase-v1.3.7-installer-verification.json`；产品源码基于 `11d545f`，测试适配及同步记录提交为 `3035bb1`。
