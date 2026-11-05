# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent è una piattaforma open source basata su Python per orchestrare agenti IA. Combina un ciclo di esecuzione autonomo, agenti specializzati, gestione persistente delle attività, esecuzione supervisionata degli strumenti e integrazioni estensibili. Tu definisci l'obiettivo; Mana-Agent aiuta a raggiungerlo usando strumenti reali e verificando i risultati.

## Funzionalità

Analizza repository, documenti e API; pianifica attività, coordina agenti, segue i progressi, riutilizza il contesto autorizzato e verifica i risultati. Supporta ricerca, programmazione, debug e automazione, oltre all'uso di browser, Git, SSH, email e Telegram.

## Avvio rapido

Richiede Python 3.10–3.14. Installazione consigliata:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Oppure dal repository:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Configura provider, credenziali e preferenze del modello con `mana-agent --configure`. Avvia `mana-agent` nel progetto. Per la dashboard facoltativa: `mana-agent dashboard --root-dir .`.

## Licenza e informazioni

[Licenza MIT](LICENSE). [README completo in inglese](README.md), [documentazione](docs/) e [segnala un problema](https://github.com/manaplatform/mana-agent/issues).
