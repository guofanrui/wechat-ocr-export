# wechat-ocr-export

把 **本机微信** 里的一段聊天，整理成可按日期阅读、可全文搜索的 Markdown / HTML。

全程只做三件事：**截图 → 系统 OCR → 拼接排版**。
不解密数据库、不改微信、不关 SIP、不联网、不需要任何账号，也不调用任何 AI 服务。

> Export a WeChat conversation on macOS into a readable, searchable Markdown/HTML
> document — by screenshotting the chat window and running Apple's Vision OCR.
> No decryption, no WeChat modification, no network, no AI service.

---

## 它长什么样

输出是两样东西：

- `outputs/微信-<对方>-聊天记录.md`：按日期分段的纯文本，方便备份、检索、喂给别的工具
- `outputs/微信-<对方>-聊天记录.html`：双栏气泡页面，浏览器里 `⌘F` 能搜任意关键词

对话里的图片会被裁出来放进 `images/`，微信表情内嵌在文字里，引用消息会复制被引用那条的原文。

## 要求

- macOS（用到了 `screencapture`、CoreGraphics、Vision，暂无 Windows/Linux 版本）
- Python 3.9+，`numpy`、`Pillow`
- Xcode Command Line Tools（`clang`、`swiftc`，用来编译三个几十行的小工具）
- 微信 Mac 版，且**聊天记录已经在本机**（工具不碰手机里的记录）

## 安装

```bash
git clone https://github.com/guofanrui/wechat-ocr-export.git
cd wechat-ocr-export
make                                    # 编译 wxwin / wxscroll / ocrfast
python3 -m venv .venv                   # 独立环境，避免污染系统 Python
source .venv/bin/activate
pip install -r requirements.txt
```

之后的命令都在这个 venv 里跑（提示符前面会出现 `(.venv)`）；重开终端时先
`cd wechat-ocr-export && source .venv/bin/activate`。

如果 `make` 在 `ocrfast` 那一步报 Swift/SDK 版本不匹配，`wxwin`、`wxscroll` 仍然可用；
按 [docs/troubleshooting.md](docs/troubleshooting.md) 处理，或用 `WXEXPORT_OCR` 指向现成的 OCR 程序。

然后给**运行脚本的终端**（或你用的 IDE）开两个权限：

- 系统设置 → 隐私与安全性 → **屏幕录制**
- 系统设置 → 隐私与安全性 → **辅助功能**

没有前者截不到窗口，没有后者滚不动聊天。授权后要重启一次终端。

## 体检

```bash
python3 scripts/wxexport.py doctor
```

它会逐项检查：macOS、numpy/Pillow、clang/swiftc、三个二进制、微信窗口、屏幕录制、辅助功能、表情模板。哪项没过，会直接告诉你怎么补。

## 用

1. 打开微信，点进要导出的那个会话，把窗口调到任意大小（程序会归一化到固定画布），**别动它**。
2. 采集（自动从最新消息往上滚，直到会话顶端）：

```bash
python3 scripts/wxexport.py --peer "对方的名字" --me "你的微信昵称" \
    capture --from-bottom
```

3. 合并成一条时间线：

```bash
python3 scripts/wxexport.py assemble --runs a
```

4. 生成文档：

```bash
python3 scripts/wxexport.py render --out-dir outputs
```

跑完 `outputs/` 里就是 `.md` 和 `.html`。

采集期间鼠标键盘会被程序占用（它要发滚动事件），建议先存盘、别切窗口。

### 常用参数

| 参数 | 作用 |
|---|---|
| `--peer` / `WXEXPORT_PEER` | 对方显示名，写进标题；引用里也靠它区分说话人 |
| `--me` / `WXEXPORT_ME` | 你自己的微信昵称（引用里显示为「昵称：…」）；不填会从引用里自动推断 |
| `WXEXPORT_OCR` | 指向自定义的 `ocrfast` 路径（默认用 `tools/ocrfast`） |
| `capture --stop-before 2025-07-01` | 只导出到某个日期之前 |
| `capture --max-steps 3` | 只抓 3 屏，用来先看效果 |
| `capture --tag h` | 分段采集，产物叫 `screens_h.json`，之后 `--runs h c` 可合并多段 |
| `reparse --reocr` | 改了代码后重跑 OCR + 解析（约 1.5 秒/屏） |

## 微信表情（可选）

微信自带的那 108 个表情，程序可以通过模板比对认出来并内嵌成图片。
模板必须**从你自己的微信里生成**（微信的表情美术资源版权归腾讯，本仓库不附带）：

1. 打开微信表情面板，翻页截图，存到 `emoji_panel/`
2. 运行 `python3 scripts/emoji_panel_build.py`

没有模板也能跑，只是表情会显示成 `〔表情〕` 占位。系统 emoji（🌹🙏 这类）本来就不在面板里，一律按占位处理。

## 它是怎么工作的

见 [docs/how-it-works.md](docs/how-it-works.md)。一句话版本：

> 每一屏都在**整窗坐标**里做 OCR（截图固定成同一尺寸、偏移恒为 0），
> 然后靠"气泡填色占比"区分文字气泡和图片，靠"滚动位移 + 文字锚点"跨屏拼接。

## 导出之后：想修错字怎么办（可选）

工具只保证"尽量准"：中文正文准确率高，英文、生僻字、聊天里发的截图上的小字会出错。
如果你想让结果更干净，可以把导出的 Markdown 交给任意 AI 编程助手（Codex、Claude 等），
配上 `work/` 里的截图做对照，让它逐条校对。这一步是**可选的增值步骤**，不是运行前提：

```text
这是我从微信导出的聊天记录（md），里面有 [图片] 和 [表情] 占位。
请对照 shots/ 里的截图，只修正明显的 OCR 错字，不要改写语气和内容，
把修改处列成表格给我确认。
```

## 已知限制

- OCR 有错字：中文正文准确率高，英文、生僻字、聊天里发的截图上的小字错误偏多
- 时间戳只能精确到微信显示的那几档（今天/昨天/日期+时段），同一时段的多条消息共用一个时间
- 语音只记 `[语音]`，不转写
- 撤回/已删除的消息无法恢复
- 长于一屏的消息靠上下两屏各自的一半拼接；如果某一半在任何一屏都没完整出现，会缺
- 个别表情可能被识别成汉字，或整条显示为占位

遇到问题先看 [docs/troubleshooting.md](docs/troubleshooting.md)。

## 隐私

这个工具处理的是**聊天记录截图**，属于最敏感的数据之一。仓库里默认忽略了
`shots/ ocr/ build/ images/ outputs/ emoji_panel/ emoji_templates/`，请不要用 `git add -f` 把它们塞进去。
交付前建议确认一遍：

```bash
git ls-files | grep -E 'shots/|ocr/|outputs/|\.png'   # 应该没有任何输出
```

## 合规

- 只导出**你自己**参与、且对方知情的对话；不要用它监视他人
- 不要上传他人的聊天内容到公开仓库或云端
- 本工具不破解微信，只是读屏幕；请自行确认符合当地法律与微信用户协议

## License

MIT（见 [LICENSE](LICENSE)）。
