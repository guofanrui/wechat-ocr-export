# 常见问题

## `make` 在编译 ocrfast 时报 Swift / SDK 版本不匹配

典型信息：

```
failed to build module 'Swift'; this SDK is not supported by the compiler
(SDK built with Apple Swift 6.3.2, compiler is 6.3.3)
```

这是本机 Command Line Tools 与 SDK 版本对不上，和代码无关。按顺序试：

1. 更新工具链：`sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install`
2. 装了完整 Xcode 的话，切过去：`sudo xcode-select -s /Applications/Xcode.app && sudo xcodebuild -runFirstLaunch`
3. 先别编，复用已有二进制：

```bash
export WXEXPORT_OCR=/path/to/ocrfast     # 任意可用的 Vision OCR 可执行文件
python3 scripts/wxexport.py doctor
```

`wxwin` / `wxscroll` 只用 clang，上面第 1 步之前一般就能编出来。

## doctor 说「屏幕录制权限」没过

- 系统设置 → 隐私与安全性 → 屏幕录制 → 勾选**你实际运行命令的那个程序**（Terminal / iTerm / VS Code / Codex）
- 勾选后**完全退出**再重开，macOS 只在进程启动时检查
- 公司电脑如果被 MDM 管控，可能需要在描述文件里放行

## doctor 说「辅助功能权限」没过

- 系统设置 → 隐私与安全性 → 辅助功能 → 勾选同一个程序
- 也可以用 `tools/wxscroll --check` 单独验证，返回 0 表示已授权

## 采集到的截图是黑的

同样是屏幕录制权限。注意：用 A 终端授权，却用 B 终端跑命令，是不生效的。

## 采集时微信没动 / 只抓了一屏

- 微信窗口必须在最前面：脚本会 `open -a WeChat`，但如果你手动切走了窗口，滚动会打到别的程序上
- 屏幕缩放或"更多空间"分辨率下窗口可能很小，先把微信窗口拉大再试
- `capture --max-steps` 默认 200，如果历史很长会跑很久，可以先用 `--stop-before 2025-01-01` 限制

## 输出的日期乱了 / 少了一段

- 先确认采集时有没有漏屏：`ls shots/ | wc -l`
- 重新合并不会重新 OCR：`python3 scripts/wxexport.py assemble --runs a`
- 如果换了微信版本、气泡颜色变了，`wxparse.py` 顶部的 `OUT_FILL` / `IN_FILL` 需要按新截图取样

## 表情全是 `〔表情〕`

没生成模板。见 README「微信表情」：用自己的表情面板截图 → `emoji_panel_build.py`。
系统 emoji（🌹🙏💀）不在微信面板里，本来就不会被识别成微信表情。

## 想只重跑某一屏

```bash
python3 scripts/wxexport.py reparse --runs a        # 复用已有 OCR
python3 scripts/wxexport.py reparse --runs a --reocr  # 连 OCR 一起重跑
```
