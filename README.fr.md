# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent est une plateforme open source en Python pour orchestrer des agents IA. Elle réunit une boucle d'exécution autonome, des agents spécialisés, une gestion persistante des tâches, l'exécution supervisée d'outils et des intégrations extensibles. Vous fixez l'objectif ; Mana-Agent aide à l'atteindre avec de vrais outils et vérifie les résultats.

## Capacités

Analyser des dépôts, des documents et des API ; planifier les tâches, coordonner les agents, suivre les progrès, réutiliser le contexte autorisé et vérifier les résultats. La plateforme aide à la recherche, au développement, au débogage et à l'automatisation, notamment avec le navigateur, Git, SSH, le courriel et Telegram.

## Démarrage rapide

Python 3.10–3.14 est requis. Installation recommandée :

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Ou depuis le dépôt :

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Configurez le fournisseur de modèle, les identifiants et les préférences avec `mana-agent --configure`. Lancez `mana-agent` dans votre projet. Pour le tableau de bord facultatif : `mana-agent dashboard --root-dir .`.

## Licence et informations

[Licence MIT](LICENSE). [README anglais complet](README.md), [documentation](docs/) et [signaler un problème](https://github.com/manaplatform/mana-agent/issues).
