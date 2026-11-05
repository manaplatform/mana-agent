# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent 是以 Python 建立的開源 AI 代理協作平台，結合自主執行迴圈、專業代理、持久任務管理、受監督的工具執行及可擴充整合。你設定目標，Mana-Agent 使用真實工具並檢查結果，協助你完成任務。

## 功能

分析程式碼儲存庫、文件與 API；規劃任務、協調代理、追蹤進度、重用已授權的上下文並驗證結果。平台支援研究、程式開發、除錯與自動化，也可與瀏覽器、Git、SSH、電子郵件和 Telegram 協作。

## 快速開始

需要 Python 3.10–3.14。建議的安裝方式：

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

或從儲存庫安裝：

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

使用 `mana-agent --configure` 設定模型供應商、憑證及模型偏好。在專案目錄執行 `mana-agent`。選用的儀表板可透過 `mana-agent dashboard --root-dir .` 啟動。

## 授權與更多資訊

[MIT 授權](LICENSE) · [完整英文 README](README.md) · [文件](docs/) · [回報問題](https://github.com/manaplatform/mana-agent/issues)
