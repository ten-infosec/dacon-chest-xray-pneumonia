# 🫁 흉부 X-ray 폐렴 분류 — DACON 해커톤

> 오즈코딩 [초격차] AI 헬스케어 딥러닝 트랙 · 개인 과제 · 2026.10.05 ~ 10.18
> 소아 흉부 X-ray로 **정상 / 폐렴**을 분류하는 이진 분류 문제

| | Baseline | 최종 (E1b) |
|---|---|---|
| **리더보드 Accuracy** | 0.7468 | **0.9407** (+19.4%p) |
| 검증 AUC | 0.9832 | 0.9979 |
| 놓친 폐렴 (FN) | 44 | **18** |
| 오진한 정상 (FP) | 13 | **3** |

## 한 문장 요약

연습 문제(학습 이미지)와 시험 문제(테스트 이미지)의 **생김새가 달라서 생긴 성능 격차**를 찾아내고, 모델이 쓰는 **"커닝 단서"를 하나씩 확인**해 가며 리더보드를 0.75에서 0.94로 올렸다.

## 핵심 발견

1. **학습·테스트 분포 차이** — Baseline은 검증 0.95 / 리더보드 0.75. 학습 이미지는 검은 패딩이 붙은 224×224, 테스트는 여백 없는 원본 크기였다.
2. **전체 미세조정이 핵심** — ImageNet 사전학습 ResNet18을 분류층만 학습하면 0.83, 모든 층을 학습하면 0.94.
3. **검증이 좋아진 실험이 전부 리더보드에서 하락** — Crop(E2), 클래스 가중치 손실(E4), 코사인 스케줄러(E5). Crop 후에는 가로세로 비율이 새로운 지름길 단서가 됐다.
4. **Grad-CAM으로 지름길 학습 확인** — 직접 구현한 Grad-CAM과 "테두리 집중 비율" 지표로, 오진 사례가 폐가 아니라 검은 모서리를 보고 있음을 확인했다 (FP 0.68 vs 맞힌 정상 0.21).
5. **재현성** — 최종 파이프라인을 처음부터 다시 실행해 제출 파일과 **624장 중 0장 차이**.

## 실험 요약

| 실험 | 바꾼 것 | Val AUC | FN | FP | 리더보드 | 판단 |
|---|---|---|---|---|---|---|
| E0 | Baseline SimpleCNN 128 | 0.9832 | 44 | 13 | 0.7468 | 기준 |
| E1a | ResNet18, 분류층만 학습 | 0.9944 | 10 | 27 | 0.8333 | |
| **E1b** | **ResNet18, 전체 미세조정** | **0.9979** | **18** | **3** | **0.9407** | **최종** |
| E2 | + 검은 여백 Crop | 0.9988 | 11 | 2 | 0.9022 | |
| E2b | + Crop + 고정 비율 | 0.9967 | 24 | 6 | 0.8990 | |
| E4 | + 클래스 가중치 손실 | 0.9984 | 31 | 2 | 0.9327 | |
| E5 | + 코사인 학습률 스케줄러 | 0.9983 | 9 | 3 | 0.9375 | |

전체 수치: [`05_Results/results.csv`](05_Results/results.csv) · 상세 분석: [`04_Report/Final_Report.md`](04_Report/Final_Report.md)

## 폴더 구조

```
├── 01_Daily_Log/     데일리 로그 (Day01 ~ Day05)
├── 02_Notebooks/     01_EDA → 02_Baseline → 03_Experiments → 04_Analysis_GradCAM, Final_Pipeline (최종 제출 코드)
├── 03_Figures/       분포·혼동행렬·임계값 그래프
├── 04_Report/        Final_Report (최종 보고서), Key_Findings, Mission0_Strategy, Baseline 분석
├── 05_Results/       results.csv (전체 실험 기록), submissions/ (리더보드 제출 파일)
└── 06_Study/         용어집 · 설명 대본 · 예상 질문 (복습용)
```

## 환경

- Google Colab (T4 GPU), PyTorch, torchvision ResNet18 (ImageNet 사전학습 가중치)
- Seed 42 고정, 층화 분할 8:2, cuDNN deterministic

## 참고 사항

- **대회 데이터와 X-ray 이미지는 포함하지 않았습니다.** 대회 데이터 재배포 금지 규칙에 따라 노트북의 X-ray 출력(샘플·오답·Grad-CAM 그림)은 제거했고, 해석은 보고서에 글과 수치로 정리했습니다.
- 노트북을 실행하려면 대회 데이터(`open.zip`)를 구글 드라이브 `dacon_chest_xray/data/`에 두어야 합니다.
- 대회 페이지: https://dacon.io/competitions/official/236593/overview/description
- Grad-CAM: Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
