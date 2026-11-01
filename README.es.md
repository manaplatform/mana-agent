# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>
[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent es una plataforma de código abierto basada en Python para coordinar agentes de IA. Combina un ciclo de ejecución autónoma, agentes especializados, gestión persistente de tareas, ejecución supervisada de herramientas e integraciones extensibles. Tú defines el objetivo; Mana-Agent ayuda a alcanzarlo con herramientas reales y comprobación de resultados.

## Capacidades

Analiza repositorios, documentos y API; planifica tareas, coordina agentes, sigue el progreso, reutiliza contexto autorizado y verifica resultados. Ayuda con investigación, programación, depuración y automatización, además de trabajar con navegador, Git, SSH, correo y Telegram.

**Dirección en vivo:** todavía no se puede dirigir una ejecución en curso, tampoco con Ctrl+S. Plan Mode sigue siendo independiente y mantiene sus requisitos de aprobación. Consulta el [diseño de dirección](docs/live-steering-design.md).

OpenAI Batch se utiliza solo a petición expresa; los demás lotes usan la ejecución paralela interna. Las tareas que requieren Batch se suspenden de forma persistente y se reanudan al recibir los resultados; los seguimientos conservan el contexto del usuario y del asistente entre interfaces. [Ejecución por lotes y recuperación](docs/26-openai-batch-api.md).

## Inicio rápido

Requiere Python 3.10–3.14. Instalación recomendada:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

O desde el repositorio:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Configura el proveedor, las credenciales y las preferencias del modelo con `mana-agent --configure`. Ejecuta `mana-agent` en tu proyecto. Para el panel opcional usa `mana-agent dashboard --root-dir .`.

## Licencia y más información

[Licencia MIT](LICENSE). [README completo en inglés](README.md), [documentación](docs/) y [notificar un problema](https://github.com/manaplatform/mana-agent/issues).
