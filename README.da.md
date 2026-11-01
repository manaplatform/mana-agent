# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>
[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent er en open source-platform i Python til koordinering af AI-agenter. Den samler en autonom eksekveringsløkke, specialiserede agenter, vedvarende opgavestyring, overvåget værktøjsudførelse og udvidelige integrationer. Du sætter målet; Mana-Agent hjælper med at nå det ved hjælp af rigtige værktøjer og kontrol af resultater.

## Muligheder

Analyser repositories, dokumenter og API'er; planlæg opgaver, koordiner agenter, følg fremskridt, genbrug godkendt kontekst og verificer resultater. Platformen understøtter research, programmering, fejlfinding og automatisering samt arbejde med browser, Git, SSH, e-mail og Telegram.

**Live-styring:** Styring af en igangværende kørsel, herunder med Ctrl+S, er endnu ikke tilgængelig. Plan Mode er adskilt, og kravene til godkendelse gælder fortsat. Se [designet for styring](docs/live-steering-design.md).

OpenAI Batch bruges kun efter udtrykkelig anmodning; anden batchbehandling bruger intern parallel eksekvering. Opgaver, der kræver Batch, sættes varigt på pause og genoptages med resultaterne; opfølgninger bevarer brugerens og assistentens kontekst på tværs af grænseflader. [Batch og gendannelse](docs/26-openai-batch-api.md).

## Hurtig start

Kræver Python 3.10–3.14. Anbefalet installation:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Eller fra repositoriet:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Konfigurer modeludbyder, legitimationsoplysninger og præferencer med `mana-agent --configure`. Kør `mana-agent` i dit projekt. Det valgfrie dashboard startes med `mana-agent dashboard --root-dir .`.

## Licens og mere information

[MIT-licens](LICENSE). [Komplet engelsk README](README.md), [dokumentation](docs/) og [rapportér et problem](https://github.com/manaplatform/mana-agent/issues).
