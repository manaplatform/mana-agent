# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>
[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent — платформа с открытым исходным кодом на Python для координации ИИ-агентов. Она объединяет автономный цикл выполнения, специализированных агентов, постоянное управление задачами, контролируемый запуск инструментов и расширяемые интеграции. Вы задаёте цель, а Mana-Agent помогает достичь её с помощью реальных инструментов и проверки результатов.

## Возможности

Анализ репозиториев, документов и API; планирование задач, координация агентов, отслеживание прогресса, повторное использование разрешённого контекста и проверка результатов. Платформа помогает в исследованиях, разработке, отладке и автоматизации, а также работе с браузером, Git, SSH, почтой и Telegram.

**Управление в реальном времени:** управление текущим выполнением, в том числе через Ctrl+S, пока недоступно. Plan Mode остаётся отдельным режимом, и его требования к утверждению сохраняются. См. [проект управления](docs/live-steering-design.md).

OpenAI Batch используется только по явному запросу; другие пакетные задачи используют внутреннее параллельное выполнение. Задачи, которым нужен Batch, сохраняют состояние ожидания и возобновляются после получения результатов; последующие сообщения сохраняют контекст пользователя и ассистента между интерфейсами. [Пакетное выполнение и восстановление](docs/26-openai-batch-api.md).

## Быстрый старт

Требуется Python 3.10–3.14. Рекомендуемая установка:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Или из репозитория:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Настройте поставщика модели, учётные данные и предпочтения с помощью `mana-agent --configure`. Запустите `mana-agent` в каталоге проекта. Необязательная панель: `mana-agent dashboard --root-dir .`.

## Лицензия и информация

[Лицензия MIT](LICENSE). [Полный README на английском](README.md), [документация](docs/) и [сообщить о проблеме](https://github.com/manaplatform/mana-agent/issues).
