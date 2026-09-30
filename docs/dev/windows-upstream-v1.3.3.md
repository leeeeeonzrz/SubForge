# Windows 定制版同步 v1.3.3

日期：2026-09-30。上游：`henry1786580051-lang/SubForge`，`v1.3.3` / `f8427da`（2026-09-18）。升级前：`c9fa15c`，基于 v1.1.15，包含 18 个本地定制提交。

## 合入方式与回退

在 `codex/integrate-upstream-v1.3.3-windows` 合入上游，保留双方提交历史；验证后将 `codex/windows-main` 快进到该结果。代码备份为 `backup/windows-before-v1.3.3-20260930`。旧桌面目录保存在 `artifacts/SubForge-1.1.15-backup-20260930`，旧安装包仍在 `artifacts/`。本次未推送或发布 GitHub Release。

## 上游变化

- v1.1.16–v1.2.1：翻译恢复、模型适配、语音缺口恢复、字幕边界和语义检查。
- v1.3.0–v1.3.1：工作区与设置界面重做、撤销及快捷键、保存锁、文档修订管理、任务配置持久化、上传字幕的持久保存。
- v1.3.2–v1.3.3：模型下载字节数/速度/剩余时间、字幕检查列表、基于原文和词时间轴的结构边界修复、隔离的开发评估产物。

## 保留及适配的定制

| 范围 | 结果 |
| --- | --- |
| Windows 运行时 | 保留 FasterWhisper/WhisperX CUDA、CPU 回退、独立工作进程、单并发线程安全、硬件探测及工作进程隐藏窗口。 |
| 模型管理 | 保留 FasterWhisper 与 Windows WhisperX 模型共享、受管模型删除、多语言对齐和说话人模型管理。 |
| 字幕输出 | 处理完成及“保存全部”继续生成 `_original`、`_translated`、`_bilingual` 三份；原文侧文件用于再次翻译，防止译文被当成原文。 |
| 保存与预览 | 三份输出接入上游持久保存和文档保存锁；生成路径分别授权；候选流程三份文件同样隔离。旧预览策略增加修订号判断，防止只有修订号的轮询触发旧文件回读。 |
| 导出与包装 | 保留按语言模式命名、Windows 原生另存为反馈、直接导出按钮、炉子图标、中文安装脚本和 CUDA DLL 检查。导出时消除重复模式后缀。 |
| 韩语 | 保留词间空格、短主语/连接词/连接语尾/定语依附断句规则、长度控制、非英语对齐审计及六处韩语提示约束。 |

韩语边界修复接入上游新的标点辅助模块，并增加保护：明确非韩语标签、已译条目、文本与词列表不一致、非法或重叠词时间轴均不重建；正常合并保留原始词对象、时间戳和说话人。上游反思模式的本地 Qwen 专用路径继续采用其选择性修复策略。

## 验证

- 独立环境 `artifacts/venv-validation`：`pytest -m "not integration"`，2947 passed、17 skipped、35 deselected。
- 前端 Vitest：62 passed；定制 `taskPreviewPolicy`：6 passed。
- Ruff：通过；Pyright：0 errors、46 warnings（包括可选平台依赖与原有告警）。
- 前端 ESLint、生产构建及 `uv lock --check`：通过。
- 修复了上游新测试在 Windows 上的隐式 GBK 读取与重复 CRLF 写入问题。
- 本次没有调用付费翻译 API，也未以真实韩语长视频测量识别质量；单元测试通过不代表端到端语音质量已经实测。
- 旧开发环境的全套测试曾在 ONNX 原生库加载时崩溃，完整离线回归因此使用独立环境；打包运行时另行验证。

原始验证日志位于本地 `artifacts/integration-*.txt`，不提交测试数据或模型文件。

## 本地桌面交付

- 构建版本：`1.3.3-windows.1`；源码版本：`1.3.3+windows.1`。
- 启动文件：`dist/SubForge/SubForge.exe`，须保留同目录的 `_internal` 运行库。
- 最终前端重新构建后逐文件 SHA-256 比对，27 个文件与桌面包内一致。
- `scripts/smoke_desktop.py dist/SubForge` 返回 0：FFmpeg/FFprobe、FastAPI 路由及 HTTP、whisper.cpp、DeepFilterNet3 实际推理、FasterWhisper/Silero VAD、WhisperX 设备选择与 PyTorch CUDA 探针、forced alignment、pyannote 导入全部通过。
- 本次提供本地桌面目录，没有生成新的 setup 安装程序，也没有修改已安装的应用。
