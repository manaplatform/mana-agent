# Mana-Agent

[English](README.md) · [العربية](README.ar.md) · [বাংলা](README.bn.md) · [Brezhoneg](README.br.md) · [Bosanski](README.bs.md) · [Dansk](README.da.md) · [Deutsch](README.de.md) · [Español](README.es.md) · [فارسی](README.fa.md) · [Français](README.fr.md) · [Ελληνικά](README.gr.md) · [Italiano](README.it.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Norsk](README.no.md) · [Polski](README.pl.md) · [Русский](README.ru.md) · [ไทย](README.th.md) · [Türkçe](README.tr.md) · [Українська](README.uk.md) · [Tiếng Việt](README.vi.md) · [简体中文](README.zh.md) · [繁體中文](README.zht.md)

Mana-Agent는 AI 에이전트를 조율하는 Python 기반 오픈 소스 플랫폼입니다. 자율 실행 루프, 전문 에이전트, 지속적인 작업 관리, 감독되는 도구 실행 및 확장 가능한 통합 기능을 결합합니다. 사용자가 목표를 정하면 Mana-Agent가 실제 도구를 사용하고 결과를 확인하며 달성을 돕습니다.

## 기능

저장소, 문서와 API를 분석하고 작업을 계획하며 에이전트를 조율하고 진행 상황을 추적합니다. 승인된 맥락을 재사용하고 결과를 검증합니다. 조사, 개발, 디버깅, 자동화뿐 아니라 브라우저, Git, SSH, 이메일, Telegram 사용도 지원합니다.

**실시간 조정:** Ctrl+S를 포함해 진행 중인 실행을 조정하는 기능은 아직 사용할 수 없습니다. Plan Mode는 별도로 유지되며 승인 요건도 계속 적용됩니다. [조정 설계](docs/live-steering-design.md)를 참조하세요.

OpenAI Batch는 명시적으로 요청한 경우에만 사용하며 다른 일괄 처리는 내부 병렬 실행을 사용합니다. Batch가 필요한 작업은 지속적으로 대기했다가 결과가 도착하면 재개됩니다. 후속 대화는 인터페이스 간 사용자와 어시스턴트의 맥락을 유지합니다. [일괄 실행과 복구](docs/26-openai-batch-api.md).

## 빠른 시작

Python 3.10–3.14가 필요합니다. 권장 설치:

```bash
pipx install git+https://github.com/manaplatform/mana-agent.git
```

또는 저장소에서 설치:

```bash
git clone https://github.com/manaplatform/mana-agent.git
cd mana-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

`mana-agent --configure`로 모델 제공자, 인증 정보 및 환경설정을 구성합니다. 프로젝트 디렉터리에서 `mana-agent`를 실행하세요. 선택 사항인 대시보드는 `mana-agent dashboard --root-dir .`로 시작합니다.

## 라이선스 및 추가 정보

[MIT 라이선스](LICENSE). [전체 영어 README](README.md), [문서](docs/), [문제 보고](https://github.com/manaplatform/mana-agent/issues).
