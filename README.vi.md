# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent là nền tảng mã nguồn mở dựa trên Python để điều phối các tác tử AI. Nền tảng kết hợp vòng lặp thực thi tự chủ, tác tử chuyên biệt, quản lý tác vụ bền vững, thực thi công cụ có giám sát và tích hợp mở rộng. Bạn đặt mục tiêu; Mana-Agent giúp thực hiện bằng công cụ thực và kiểm tra kết quả.

## Khả năng

Phân tích kho mã, tài liệu và API; lập kế hoạch, điều phối tác tử, theo dõi tiến độ, tái sử dụng ngữ cảnh được phép và kiểm chứng kết quả. Nền tảng hỗ trợ nghiên cứu, lập trình, gỡ lỗi và tự động hóa, cũng như làm việc với trình duyệt, Git, SSH, email và Telegram.

**Điều hướng trực tiếp:** Chưa thể điều hướng một lần thực thi đang diễn ra, kể cả bằng Ctrl+S. Plan Mode vẫn tách biệt và các yêu cầu phê duyệt của chế độ này vẫn áp dụng. Xem [thiết kế điều hướng](docs/live-steering-design.md).

OpenAI Batch chỉ được sử dụng khi có yêu cầu rõ ràng; các xử lý theo lô khác dùng thực thi song song nội bộ. Tác vụ cần Batch được tạm dừng bền vững và tiếp tục khi có kết quả; các lượt tiếp theo giữ ngữ cảnh của người dùng và trợ lý trên các giao diện. [Thực thi theo lô và khôi phục](docs/26-openai-batch-api.md).

## Bắt đầu nhanh

Cần Python 3.10–3.14. Cài đặt được khuyến nghị:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

Hoặc từ kho mã:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Cấu hình nhà cung cấp mô hình, thông tin xác thực và tùy chọn bằng `mana-agent --configure`. Chạy `mana-agent` trong thư mục dự án. Bảng điều khiển tùy chọn: `mana-agent dashboard --root-dir .`.

## Giấy phép và thông tin

[Giấy phép MIT](LICENSE). [README tiếng Anh đầy đủ](README.md), [tài liệu](docs/) và [báo lỗi](https://github.com/manaplatform/mana-agent/issues).
