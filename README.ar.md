# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

منصة مفتوحة المصدر بلغة Python لتنسيق وكلاء الذكاء الاصطناعي وتنفيذ المهام. تجمع حلقة تنفيذ ذاتية، ووكلاء متخصصين، وإدارة مستمرة للمهام، وأدوات خاضعة للإشراف وتكاملات قابلة للتوسعة. تحدد الهدف ويساعدك Mana-Agent على الوصول إليه باستخدام أدوات حقيقية وفحص النتائج.

## القدرات

يفحص المشروع المستودعات والمستندات وواجهات API، ويخطط للمهام، وينسق الوكلاء، ويتابع التقدم، ويستعيد السياق المصرح به ويتحقق من النتائج. كما يساعد في البحث والبرمجة وتصحيح الأخطاء والأتمتة واستخدام المتصفح وGit وSSH والبريد وTelegram.

**حالة التوجيه المباشر:** لا يتوفر توجيه تنفيذ جارٍ، بما في ذلك Ctrl+S، بعد. يظل وضع التخطيط منفصلاً وتبقى متطلبات الموافقة سارية. راجع [تصميم التوجيه](docs/live-steering-design.md).

يُستخدم OpenAI Batch عند طلبه صراحة فقط؛ وتستخدم المعالجة المجمعة الأخرى التنفيذ المتوازي الداخلي. تتوقف المهام التي تتطلب Batch بشكل دائم وتستأنف عند وصول النتائج، وتحتفظ المتابعات بسياق المستخدم والمساعد عبر الواجهات. راجع [التنفيذ المجمع والاستعادة](docs/26-openai-batch-api.md).

## البدء السريع

يتطلب Python 3.10–3.14. التثبيت الموصى به:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

أو من المستودع:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

اضبط مقدم النموذج وبيانات الاعتماد والتفضيلات باستخدام `mana-agent --configure`، ثم انتقل إلى مشروعك وشغّل `mana-agent`. لتشغيل لوحة التحكم الاختيارية استخدم `mana-agent dashboard --root-dir .`.

## الترخيص والمزيد

المشروع مرخص بموجب [MIT](LICENSE). راجع [الدليل الإنجليزي الكامل](README.md)، و[التوثيق](docs/)، و[صفحة المشاكل](https://github.com/manaplatform/mana-agent/issues).
