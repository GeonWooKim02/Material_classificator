# 입자 스펙트럼 기반 재료 분류

266.0–1000.0 nm 구간의 스펙트럼을 입력받아 입자의 재료를 **Al, Cu, Si₃N₄, SiO₂, W** 중 하나로 분류한다. 1차원 합성곱 신경망(CNN)과 다층 퍼셉트론(MLP)이 출력한 클래스 확률을 산술평균해 최종 재료를 결정한다. 저장소에는 학습 코드, 학습·평가 데이터, 학습된 모델, Excel 파일을 행 단위로 분석하는 데스크톱 프로그램이 들어 있다.

## 저장소 구성

| 경로 | 내용 |
| --- | --- |
| [training/](training/) | 학습 스크립트와 학습·내부 검증에 사용하는 스펙트럼 |
| [data/evaluation/](data/evaluation/) | 별도의 평가용 스펙트럼과 강도 변환본 |
| [data/examples/](data/examples/) | GUI에서 한 행의 예측을 확인할 수 있는 예제 |
| [models/research/](models/research/) | 연구용 앙상블, 개별 CNN·MLP, 표준화기, 레이블 인코더 |
| [desktop/](desktop/) | 일괄 추론 GUI와 Windows 패키징용 PyInstaller 설정 |
| [desktop/dist/](desktop/dist/) | Windows 실행 파일과 함께 사용하던 앙상블 및 전처리 파일 |
| [assets/](assets/) | 프로젝트 아이콘 |

## 저장된 모델로 예측하기

Python 환경에 다음 패키지를 설치한다. GUI 실행에는 Tk를 사용할 수 있는 데스크톱 환경이 필요하다. 아래 명령은 각각 저장소 루트에서 시작한다.

```sh
python -m pip install numpy pandas openpyxl joblib scikit-learn tensorflow customtkinter
```

```sh
cd models/research
python ../../desktop/wafer.py
```

1. **파일 선택 및 일괄 분석 시작**을 누르고 [단일 스펙트럼 예제](data/examples/Test_spectrum_변환.xlsx)를 선택한다.
2. 결과 창에서 Excel 행 번호, 예측 재료, 확신도를 확인한다. 확신도는 앙상블이 예측한 클래스의 출력 확률에 100을 곱한 값이다.
3. 여러 행을 가진 파일을 선택하면 행별 결과와 재료별 예측 건수가 함께 표시된다. 정답 레이블이 있는 평가 파일에서는 예측 재료를 같은 행의 `Material` 값과 대조할 수 있다.

GUI는 스크립트 폴더를 먼저 확인하고 현재 작업 폴더에서도 모델을 찾는다. 위 명령은 연구용 모델을 선택한다. Windows 배포용 모델을 소스에서 사용하려면 저장소 루트에서 `desktop/dist`로 이동해 `python ../wafer.py`를 실행한다. 두 폴더에서 이름이 같은 `model.keras`, `scaler.pkl`, `label_encoder.pkl`은 내용이 다르므로 세 파일을 같은 폴더의 세트로 사용한다.

| 모델 세트 | 모델에 기록된 Keras 버전 | 전처리 파일에 기록된 scikit-learn 버전 |
| --- | --- | --- |
| 연구용 | `3.16.0.dev2026101604` | `1.9.1` |
| Windows 배포용 | `3.14.1` | `1.9.0` |

위 버전은 저장 시점의 메타데이터다. 원 실험의 전체 패키지 버전을 고정한 환경 파일은 제공되지 않는다.

## 데이터와 입력 형식

각 통합 문서의 첫 시트에서 첫 행은 헤더이며, 이후 한 행이 스펙트럼 하나에 해당한다. **앞 500개 열**은 `266.0nm`에서 `1000.0nm`까지 파장 순서로 배열한 수치다. 프로그램은 열 위치로 입력을 읽으므로 새 파일도 제공 예제의 열 순서를 유지해야 한다. 추가 정보는 500개 측정값 뒤에 둔다. GUI는 완전히 빈 행을 건너뛰고, 측정값에 빈칸이나 숫자로 변환할 수 없는 값이 있는 행은 오류로 표시한다.

| 파일 | 데이터 행 | 용도 |
| --- | ---: | --- |
| [particle_dataset_all_materials.xlsx](training/particle_dataset_all_materials.xlsx) | 4,220 | 재료당 844개. 학습 스크립트가 형상과 크기 조건으로 선별한 뒤 학습·검증 분할 |
| [particle_testset_all_materials.xlsx](data/evaluation/particle_testset_all_materials.xlsx) | 290 | 재료당 58개. 학습 스크립트와 별도로 예측 결과를 확인하는 데이터 |
| [scaled/](data/evaluation/scaled/)의 파일 6개 | 각 290 | 강도 배율 조건 5종(×2, ×0.5, ×10, ×0.1, ×100)과 최댓값 정규화 조건(`div_max`) |
| [Test_spectrum_변환.xlsx](data/examples/Test_spectrum_변환.xlsx) | 1 | GUI 입력 예제 |

Excel과 GUI에서 질화규소와 이산화규소의 레이블은 각각 `Si3N4`, `SiO2`로 표기된다.

## 학습 방법

1. **선별 및 분할** — `Shape`가 `Cube`이면서 `Radius_X(nm)`가 30 또는 45인 행 10개를 제외한다. 남은 4,210개를 재료별 비율을 유지해 학습 3,578개와 검증 632개로 분할한다(`random_state=0`).
2. **스펙트럼 변환** — 강도에 하한값 `1e-30`을 적용하고 상용로그를 취한 뒤, 각 스펙트럼에서 해당 행의 평균 로그 강도를 뺀다. 하한값이 적용되지 않는 양의 스펙트럼에서는 전체 강도를 일정한 양의 배수로 바꾸어도 중심화된 값이 같다.
3. **증강 및 표준화** — 학습 스펙트럼마다 채널별 곱셈 잡음(배율의 평균 1, 표준편차 0.03, 하한 0.05)과 행별 강도 배율(0.5–2.0, 로그 균등분포)을 적용하고 다시 중심화한다. 이때 행 전체에 공통인 로그 배율은 중심화 과정에서 상쇄된다. 원본과 증강본 8세트를 합쳐 채널별 `StandardScaler`를 적합한다. 검증에는 한 번 증강한 검증 스펙트럼과 학습 세트에서 적합한 표준화기를 사용한다.
4. **모델 학습** — CNN은 커널 크기 5의 합성곱 층 두 개(필터 64, 128)와 128유닛 은닉층을 사용한다. MLP는 256 및 128유닛 은닉층에 각각 드롭아웃 0.3을 적용한다. 두 모델을 Adam과 희소 범주형 교차 엔트로피로 각각 학습한다. 배치 크기는 64, 최대 학습 횟수는 200 에폭이다. 검증 손실이 10 에폭 동안 개선되지 않으면 조기 종료하고 최적 가중치를 복원한다.
5. **앙상블 추론** — 새 스펙트럼에 로그 변환, 행 중심화, 저장된 표준화기를 적용한다. CNN과 MLP의 softmax 출력을 같은 가중치로 평균하고, 가장 큰 값을 갖는 클래스를 반환한다.

### 재학습 실행

학습 스크립트는 현재 작업 폴더에서 `particle_dataset_all_materials.xlsx`를 읽는다. 저장소 루트에서 스크립트와 데이터가 함께 있는 `training/`으로 이동한 뒤 실행한다.

```sh
cd training
python material_classifier_v3.py
```

학습 결과는 `training/`에 `cnn.keras`, `mlp.keras`, `model.keras`, `scaler.pkl`, `label_encoder.pkl`로 저장된다. 다시 실행하면 같은 이름의 결과를 덮어쓴다. 재학습한 앙상블로 예측하려면 같은 폴더에서 `python ../desktop/wafer.py`를 실행한다.

학습 코드는 NumPy 난수 생성기와 TensorFlow 시드를 0으로 설정한다. 학습 중 출력되는 정확도는 내부 검증 분할의 지표이며, 별도 평가 파일의 정확도는 스크립트가 자동으로 계산하지 않는다.
