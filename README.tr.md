# Mana-Agent

<p align="center">
  <img src="https://raw.githubusercontent.com/manaplatform/mana-agent/refs/heads/main/logo.png" alt="Mana-Agent" width="100%" />
</p>

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent, yapay zekâ ajanlarını koordine etmek için Python tabanlı açık kaynaklı bir platformdur. Otonom yürütme döngüsünü, uzman ajanları, kalıcı görev yönetimini, gözetimli araç yürütmesini ve genişletilebilir entegrasyonları birleştirir. Hedefi siz belirlersiniz; Mana-Agent gerçek araçlar ve sonuç kontrolüyle hedefe ulaşmanıza yardımcı olur.

## Yetenekler

Depoları, belgeleri ve API'leri analiz eder; görevleri planlar, ajanları koordine eder, ilerlemeyi izler, izinli bağlamı yeniden kullanır ve sonuçları doğrular. Araştırma, yazılım geliştirme, hata ayıklama ve otomasyonun yanı sıra tarayıcı, Git, SSH, e-posta ve Telegram kullanımına yardımcı olur.

**Canlı yönlendirme:** Devam eden bir yürütmeyi yönlendirme, Ctrl+S dahil, henüz kullanılamıyor. Plan Mode ayrı kalır ve onay gereksinimleri geçerliliğini korur. [Yönlendirme tasarımına](docs/live-steering-design.md) bakın.

OpenAI Batch yalnızca açıkça talep edildiğinde kullanılır; diğer toplu işlemler dahili paralel yürütmeyi kullanır. Batch gerektiren görevler kalıcı olarak bekletilir ve sonuçlarla devam eder; takip mesajları kullanıcı ve asistan bağlamını arayüzler arasında korur. [Toplu yürütme ve kurtarma](docs/26-openai-batch-api.md).

## Hızlı başlangıç

Python 3.10–3.14 gereklidir. Önerilen kurulum:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Veya depodan:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Model sağlayıcısını, kimlik bilgilerini ve tercihleri `mana-agent --configure` ile ayarlayın. Proje dizininde `mana-agent` çalıştırın. İsteğe bağlı kontrol paneli için `mana-agent dashboard --root-dir .` kullanın.

## Lisans ve bilgi

[MIT lisansı](LICENSE). [Tam İngilizce README](README.md), [belgeler](docs/) ve [sorun bildirme](https://github.com/manaplatform/mana-agent/issues).
