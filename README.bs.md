# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent je platforma otvorenog koda zasnovana na Pythonu za koordinaciju AI agenata. Objedinjuje autonomnu petlju izvršavanja, specijalizirane agente, trajno upravljanje zadacima, nadzirano izvršavanje alata i proširive integracije. Vi određujete cilj, a Mana-Agent pomaže da se on ostvari uz stvarne alate i provjeru rezultata.

## Mogućnosti

Analizirajte repozitorije, dokumente i API-je; planirajte zadatke, koordinirajte agente, pratite napredak, ponovo koristite odobreni kontekst i provjeravajte rezultate. Platforma pomaže pri istraživanju, programiranju, otklanjanju grešaka i automatizaciji te radu s preglednikom, Gitom, SSH-om, e-poštom i Telegramom.

## Brzi početak

Potreban je Python 3.10–3.14. Preporučena instalacija:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Ili iz repozitorija:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Podesite pružaoca modela, pristupne podatke i postavke pomoću `mana-agent --configure`. U direktoriju projekta pokrenite `mana-agent`. Opcionalna kontrolna ploča: `mana-agent dashboard --root-dir .`.

## Licenca i dodatne informacije

[MIT licenca](LICENSE). [Potpuni engleski README](README.md), [dokumentacija](docs/) i [prijava problema](https://github.com/manaplatform/mana-agent/issues).
