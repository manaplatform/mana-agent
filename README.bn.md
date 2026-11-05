# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent হল Python-ভিত্তিক উন্মুক্ত এআই এজেন্ট সমন্বয় প্ল্যাটফর্ম। এতে স্বয়ংক্রিয় এক্সিকিউশন লুপ, বিশেষজ্ঞ এজেন্ট, স্থায়ী টাস্ক ব্যবস্থাপনা, তত্ত্বাবধানে টুল চালানো এবং সম্প্রসারণযোগ্য ইন্টিগ্রেশন রয়েছে। আপনি লক্ষ্য নির্ধারণ করেন; Mana-Agent বাস্তব টুল ব্যবহার ও ফল যাচাই করে এগোতে সাহায্য করে।

## সক্ষমতা

রিপোজিটরি, নথি ও API বিশ্লেষণ; কাজের পরিকল্পনা; এজেন্ট সমন্বয়; অগ্রগতি অনুসরণ; অনুমোদিত প্রসঙ্গ পুনর্ব্যবহার এবং ফল যাচাই। গবেষণা, কোড লেখা, ডিবাগিং ও স্বয়ংক্রিয় কাজের পাশাপাশি ব্রাউজার, Git, SSH, ইমেইল এবং Telegram ব্যবহারেও সাহায্য করে।

## দ্রুত শুরু

Python 3.10–3.14 প্রয়োজন। প্রস্তাবিত ইনস্টলেশন:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

অথবা রিপোজিটরি থেকে:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`mana-agent --configure` দিয়ে মডেল প্রদানকারী, শংসাপত্র ও পছন্দ নির্ধারণ করুন। প্রকল্পের ডিরেক্টরিতে `mana-agent` চালান। ঐচ্ছিক ড্যাশবোর্ডের জন্য `mana-agent dashboard --root-dir .` ব্যবহার করুন।

## লাইসেন্স ও আরও তথ্য

[MIT লাইসেন্স](LICENSE)। [সম্পূর্ণ ইংরেজি README](README.md), [ডকুমেন্টেশন](docs/) এবং [সমস্যা জানান](https://github.com/manaplatform/mana-agent/issues)।
