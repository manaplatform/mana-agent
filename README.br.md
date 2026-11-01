# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Ur bladenn digor diazezet war Python eo Mana-Agent evit aozañ ajanted skiant artificiel ha kas labourioù da benn. Kinnig a ra ur c'helc'h erounit emren, ajanted arbennikaet, merañ padus an trevelloù, ostilhoù dindan evezh hag etrekewerioù ledanadus. Lakaat a rit ur pal; Mana-Agent a skoazell ac'hanoc'h da vont war-raok ha da wiriañ an disoc'hoù.

## Galloud

Gant Mana-Agent e c'hallit enklask er mirdioù, an teuliadur hag an APIoù, prientiñ labourioù, aozañ ajanted, heuliañ ar mont war-raok, adimplijout ar c'hemperzh aotreet ha gwiriañ an disoc'hoù. Emañ ar c'hoderezh, an enklask, an difaziañ, an emgefreekaat hag implij ar merdeer, Git, SSH, postel ha Telegram e-touez an implijoù.

**Sturerezh buhez:** n'haller ket c'hoazh sturiañ un erounit en cours, na gant Ctrl+S kennebeut. Disparti eo ar mod Plan ha dalc'het eo e ezhommoù aotre. Sellit ouzh [an tresadenn sturerezh](docs/live-steering-design.md).

Implijet eo OpenAI Batch nemet pa vez goulennet en un doare sklaer; an dasparzhioù all a implij an erounit kenstur diabarzh. An trevelloù a c'houlenn Batch a chom war c'hortoz en un doare padus ha kenderc'hel a reont gant an disoc'hoù; miret eo kemperzh an implijer hag ar skoazeller dre an etrefasoù. [Batch ha dizober](docs/26-openai-batch-api.md).

## Kregiñ prim

Ret eo Python 3.10–3.14. Staliañ erbedet:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Pe adalek ar mirdi:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Kefluniit ar pourvezer model, an titouroù anaout hag ar gwellvezioù gant `mana-agent --configure`. En ho raktres, loc'hit `mana-agent`. Evit an daolenn-stur diret: `mana-agent dashboard --root-dir .`.

## Aotre-implij ha muioc'h

[Aotre MIT](LICENSE). [README saoznek klok](README.md), [teuliadur](docs/) ha [disklêriañ ur gudenn](https://github.com/manaplatform/mana-agent/issues).
