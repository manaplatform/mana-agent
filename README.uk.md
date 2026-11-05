# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent — платформа з відкритим кодом на Python для координації агентів ШІ. Вона поєднує автономний цикл виконання, спеціалізованих агентів, постійне керування завданнями, контрольоване виконання інструментів і розширювані інтеграції. Ви визначаєте мету, а Mana-Agent допомагає її досягти за допомогою справжніх інструментів і перевірки результатів.

## Можливості

Аналіз репозиторіїв, документів і API; планування завдань, координація агентів, відстеження поступу, повторне використання дозволеного контексту та перевірка результатів. Платформа допомагає в дослідженні, розробці, налагодженні й автоматизації, а також роботі з браузером, Git, SSH, поштою та Telegram.

## Швидкий початок

Потрібен Python 3.10–3.14. Рекомендоване встановлення:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Або з репозиторію:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Налаштуйте постачальника моделі, облікові дані та параметри за допомогою `mana-agent --configure`. Запустіть `mana-agent` у каталозі проєкту. Необов'язкова панель: `mana-agent dashboard --root-dir .`.

## Ліцензія та інформація

[Ліцензія MIT](LICENSE). [Повний README англійською](README.md), [документація](docs/) та [повідомити про проблему](https://github.com/manaplatform/mana-agent/issues).
