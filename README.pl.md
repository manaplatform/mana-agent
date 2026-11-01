# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent to platforma open source w Pythonie do koordynowania agentów AI. Łączy autonomiczną pętlę wykonawczą, wyspecjalizowanych agentów, trwałe zarządzanie zadaniami, nadzorowane wykonywanie narzędzi i rozszerzalne integracje. Użytkownik określa cel, a Mana-Agent pomaga go osiągnąć za pomocą rzeczywistych narzędzi i weryfikacji wyników.

## Możliwości

Analiza repozytoriów, dokumentów i API; planowanie zadań, koordynacja agentów, śledzenie postępów, ponowne wykorzystanie dozwolonego kontekstu i weryfikacja wyników. Platforma pomaga w badaniach, programowaniu, debugowaniu i automatyzacji oraz obsłudze przeglądarki, Git, SSH, poczty i Telegrama.

**Sterowanie na żywo:** kierowanie trwającym wykonaniem, również przez Ctrl+S, nie jest jeszcze dostępne. Plan Mode pozostaje osobny, a jego wymagania zatwierdzania nadal obowiązują. Zobacz [projekt sterowania](docs/live-steering-design.md).

OpenAI Batch jest używany tylko na wyraźne żądanie; inne przetwarzanie wsadowe korzysta z wewnętrznego wykonywania równoległego. Zadania wymagające Batch są trwale wstrzymywane i wznawiane po otrzymaniu wyników; kontynuacje zachowują kontekst użytkownika i asystenta między interfejsami. [Batch i odzyskiwanie](docs/26-openai-batch-api.md).

## Szybki start

Wymagany jest Python 3.10–3.14. Zalecana instalacja:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Lub z repozytorium:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Skonfiguruj dostawcę modelu, dane uwierzytelniające i preferencje poleceniem `mana-agent --configure`. Uruchom `mana-agent` w katalogu projektu. Opcjonalny panel: `mana-agent dashboard --root-dir .`.

## Licencja i informacje

[Licencja MIT](LICENSE). [Pełny angielski README](README.md), [dokumentacja](docs/) i [zgłaszanie problemów](https://github.com/manaplatform/mana-agent/issues).
