# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent เป็นแพลตฟอร์มโอเพนซอร์สที่ใช้ Python สำหรับประสานงานเอเจนต์ AI โดยรวมวงรอบการทำงานอัตโนมัติ เอเจนต์เฉพาะทาง การจัดการงานแบบคงสถานะ การใช้เครื่องมือภายใต้การกำกับดูแล และการเชื่อมต่อที่ขยายได้ คุณกำหนดเป้าหมาย แล้ว Mana-Agent ช่วยดำเนินงานด้วยเครื่องมือจริงและตรวจสอบผลลัพธ์

## ความสามารถ

วิเคราะห์รีโพซิทอรี เอกสาร และ API วางแผนงาน ประสานเอเจนต์ ติดตามความคืบหน้า ใช้บริบทที่ได้รับอนุญาตซ้ำ และตรวจสอบผลลัพธ์ รองรับการค้นคว้า เขียนโปรแกรม แก้บั๊ก และทำงานอัตโนมัติ รวมถึงการใช้เบราว์เซอร์ Git SSH อีเมล และ Telegram

**การปรับทิศทางระหว่างทำงาน:** ยังไม่สามารถปรับทิศทางงานที่กำลังทำอยู่ รวมถึงการใช้ Ctrl+S ได้ Plan Mode แยกจากกันและยังต้องปฏิบัติตามข้อกำหนดการอนุมัติ ดู [การออกแบบการปรับทิศทาง](docs/live-steering-design.md)

ใช้ OpenAI Batch เฉพาะเมื่อร้องขออย่างชัดเจน ส่วนงานชุดอื่นใช้การทำงานแบบขนานภายใน งานที่ต้องใช้ Batch จะพักแบบคงสถานะและทำต่อเมื่อได้รับผลลัพธ์ ข้อความติดตามจะรักษาบริบทของผู้ใช้และผู้ช่วยข้ามอินเทอร์เฟซ ดู [การทำงานแบบชุดและการกู้คืน](docs/26-openai-batch-api.md)

## เริ่มต้นอย่างรวดเร็ว

ต้องใช้ Python 3.10–3.14 แนะนำให้ติดตั้งด้วย:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

หรือจากรีโพซิทอรี:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

ตั้งค่าผู้ให้บริการโมเดล ข้อมูลรับรอง และค่ากำหนดด้วย `mana-agent --configure` จากนั้นรัน `mana-agent` ในโครงการ สำหรับแดชบอร์ดซึ่งเป็นตัวเลือก ใช้ `mana-agent dashboard --root-dir .`

## สัญญาอนุญาตและข้อมูลเพิ่มเติม

[สัญญาอนุญาต MIT](LICENSE) · [README ภาษาอังกฤษฉบับเต็ม](README.md) · [เอกสาร](docs/) · [รายงานปัญหา](https://github.com/manaplatform/mana-agent/issues)
