# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>
[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent は、AI エージェントを連携させる Python 製のオープンソースプラットフォームです。自律実行ループ、専門エージェント、永続的なタスク管理、監督下でのツール実行、拡張可能な連携機能を組み合わせています。ユーザーが目標を定め、Mana-Agent が実際のツールを使い、結果を確認しながら達成を支援します。

## 主な機能

リポジトリ、文書、API の分析、タスクの計画、エージェントの調整、進捗の追跡、許可されたコンテキストの再利用、結果の検証を行います。調査、開発、デバッグ、自動化のほか、ブラウザー、Git、SSH、メール、Telegram との連携を支援します。

**ライブステアリング:** 実行中の処理の方向変更は Ctrl+S を含め、まだ利用できません。Plan Mode は独立しており、その承認要件は引き続き適用されます。[設計](docs/live-steering-design.md)を参照してください。

OpenAI Batch は明示的な要求がある場合のみ使用し、その他のバッチ処理には内部並列実行を使用します。Batch が必要なタスクは永続的に待機し、結果の到着後に再開します。フォローアップでは、各画面間でユーザーとアシスタントのコンテキストが保持されます。[バッチ実行と復旧](docs/26-openai-batch-api.md)。

## クイックスタート

Python 3.10–3.14 が必要です。推奨インストール方法:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

またはリポジトリから:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`mana-agent --configure` でモデルプロバイダー、認証情報、設定を構成します。プロジェクト内で `mana-agent` を実行します。任意のダッシュボードは `mana-agent dashboard --root-dir .` で起動します。

## ライセンスと詳細

[MIT ライセンス](LICENSE)。[英語の完全な README](README.md)、[ドキュメント](docs/)、[問題の報告](https://github.com/manaplatform/mana-agent/issues)。
