# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent er en åpen kildekode-plattform i Python for koordinering av KI-agenter. Den kombinerer en autonom kjøresløyfe, spesialiserte agenter, vedvarende oppgavehåndtering, overvåket verktøykjøring og utvidbare integrasjoner. Du setter målet; Mana-Agent hjelper deg dit med virkelige verktøy og kontroll av resultatene.

## Muligheter

Analyser kodearkiver, dokumenter og API-er; planlegg oppgaver, koordiner agenter, følg fremdrift, gjenbruk godkjent kontekst og kontroller resultater. Plattformen hjelper med undersøkelser, programmering, feilsøking og automatisering, samt bruk av nettleser, Git, SSH, e-post og Telegram.

**Direktestyring:** Styring av en pågående kjøring, inkludert Ctrl+S, er ennå ikke tilgjengelig. Plan Mode er separat, og kravene til godkjenning gjelder fortsatt. Se [designet for styring](docs/live-steering-design.md).

OpenAI Batch brukes bare ved uttrykkelig forespørsel; annen batchbehandling bruker intern parallell kjøring. Oppgaver som krever Batch, settes varig på vent og fortsetter når resultatene kommer; oppfølginger beholder brukerens og assistentens kontekst på tvers av grensesnitt. [Batch og gjenoppretting](docs/26-openai-batch-api.md).

## Hurtigstart

Krever Python 3.10–3.14. Anbefalt installasjon:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Eller fra kodearkivet:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Konfigurer modelltilbyder, legitimasjon og innstillinger med `mana-agent --configure`. Kjør `mana-agent` i prosjektmappen. For det valgfrie dashbordet: `mana-agent dashboard --root-dir .`.

## Lisens og mer informasjon

[MIT-lisens](LICENSE). [Fullstendig engelsk README](README.md), [dokumentasjon](docs/) og [meld et problem](https://github.com/manaplatform/mana-agent/issues).
