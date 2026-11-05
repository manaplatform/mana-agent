# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent 是一个基于 Python 的开源 AI 智能体编排平台，结合了自主执行循环、专业智能体、持久任务管理、受监督的工具执行和可扩展集成。你确定目标，Mana-Agent 使用真实工具并检查结果，帮助你完成任务。

## 功能

分析代码仓库、文档和 API；规划任务、协调智能体、跟踪进度、复用已授权的上下文并验证结果。平台支持研究、编程、调试和自动化，也可与浏览器、Git、SSH、电子邮件和 Telegram 协作。

## 快速开始

需要 Python 3.10–3.14。推荐安装方式：

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

或者从仓库安装：

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

使用 `mana-agent --configure` 配置模型提供商、凭据和模型偏好。在项目目录中运行 `mana-agent`。可选仪表板可通过 `mana-agent dashboard --root-dir .` 启动。

## 许可证与更多信息

[MIT 许可证](LICENSE) · [完整英文 README](README.md) · [文档](docs/) · [报告问题](https://github.com/manaplatform/mana-agent/issues)
