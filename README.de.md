# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent ist eine quelloffene, Python-basierte Plattform zur Koordination von KI-Agenten. Sie verbindet eine autonome Ausführungsschleife, spezialisierte Agenten, dauerhafte Aufgabenverwaltung, überwachte Werkzeugausführung und erweiterbare Integrationen. Sie geben das Ziel vor; Mana-Agent hilft mit echten Werkzeugen und überprüft die Ergebnisse.

## Fähigkeiten

Repositories, Dokumente und APIs analysieren; Aufgaben planen, Agenten koordinieren, Fortschritt verfolgen, freigegebenen Kontext wiederverwenden und Ergebnisse prüfen. Die Plattform unterstützt Recherche, Programmierung, Fehlersuche und Automatisierung sowie die Arbeit mit Browser, Git, SSH, E-Mail und Telegram.

## Schnellstart

Python 3.10–3.14 wird benötigt. Empfohlene Installation:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Oder aus dem Repository:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Konfigurieren Sie Modellanbieter, Zugangsdaten und Einstellungen mit `mana-agent --configure`. Starten Sie `mana-agent` im Projektverzeichnis. Das optionale Dashboard starten Sie mit `mana-agent dashboard --root-dir .`.

## Lizenz und weitere Informationen

[MIT-Lizenz](LICENSE). [Vollständige englische README](README.md), [Dokumentation](docs/) und [Problem melden](https://github.com/manaplatform/mana-agent/issues).
