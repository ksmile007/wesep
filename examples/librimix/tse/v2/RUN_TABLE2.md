# Table 2 재현 — 처음부터 끝까지

wesep 논문(Interspeech 2024)의 **Table 2** 를 이 서버에서 다시 만드는 방법.

Table 2 는 화자 임베딩을 분리 모델에 **어떻게 섞느냐**만 바꾼 비교임.
모델·데이터·학습 설정은 전부 같고 `spk_fuse_type` 한 줄만 다른 **네 번의 학습**임.

| 칸 | `spk_fuse_type` | config |
|---|---|---|
| 1 | `concat` | [confs/bsrnn_ecapa_concat.yaml](confs/bsrnn_ecapa_concat.yaml) |
| 2 | `additive` | [confs/bsrnn_ecapa_additive.yaml](confs/bsrnn_ecapa_additive.yaml) |
| 3 | `multiply` | [confs/bsrnn_ecapa_multiply.yaml](confs/bsrnn_ecapa_multiply.yaml) |
| 4 | `FiLM` | [confs/bsrnn_ecapa_FiLM.yaml](confs/bsrnn_ecapa_FiLM.yaml) |

> 이 문서는 **이 서버에서 실제로 쓰는 명령어**만 담음.
> wesep 일반 사용법은 [README.md](README.md) 에 있음 — 그 파일은 건드리지 않았음.

---

## 전체 흐름 — 한눈에

| 구분 | stage | 하는 일 | 걸리는 시간 | 왜 이렇게 나뉘나 |
|---|---|---|---|---|
| **공용**<br>한 번만 | **1** | Libri2Mix 를 읽어 데이터 목록 만들기 | 몇 분 | 네 칸이 **똑같은 데이터**를 씀.<br>한 번 만들어 두면 다시 안 해도 됨 |
| | **2** | shard tar 로 묶기 | 20~30분 | |
| **개별**<br>실험마다 | **3** | **학습** (150 epoch) | **약 40시간** | fusion 마다 따로 학습하고 따로 평가함.<br>`exp_dir` 이 달라야 섞이지 않음 |
| | **4** | 체크포인트 평균 | 1분 | |
| | **5** | test 셋 추론 | 25분 | |
| | **6** | 채점 (SI-SNRi · PESQ · STOI · DNSMOS) | 10분 | |

stage 1·2 의 결과는 `data/clean/{train-100,dev,test}` 이고,
stage 3~6 의 결과는 `exp/bsrnn_ecapa_<fusion>/` 입니다 —
**Table 2 에 쓸 숫자는 그 안의 `infer_utt_scores.csv`** 에 있습니다.

**GPU 는 1장당 run 1개**로 씁니다. 4장이 있으면 네 칸을 동시에 돌려 **약 40시간**에 끝납니다.

---

## 0. 환경 만들기

처음 한 번만 하면 됨.

### 1) 환경과 패키지 — 두 가지 방법 중 **하나만**

**방법 A — pip 목록으로**

```bash
cd /workspace/git_clone/SD-FiLM/wesep
conda create -n wesep2 python=3.9 -y && conda activate wesep2 && pip install -r requirements_wesep2.txt
```

**방법 B — conda 환경 파일로**

```bash
cd /workspace/git_clone/SD-FiLM/wesep
conda env create -f environment_wesep2.yaml && conda activate wesep2
```

두 파일 모두 첫 줄에 `--extra-index-url https://download.pytorch.org/whl/cu128` 이 들어 있음.
`torch==2.7.1+cu128` 은 **PyPI 에 없고** PyTorch 전용 서버에만 있어서, 그 주소가 없으면
`No matching distribution found` 로 멈춤.

### 2) `wespeaker` — PyPI 에 없으므로 git 에서

```bash
pip install git+https://github.com/wenet-e2e/wespeaker.git
```

패키지 이름이 PyPI 에 **등록돼 있지 않아**(`pip index versions wespeaker` → `from versions: none`)
1) 의 목록에는 일부러 빼 두었음. 반드시 따로 깔아야 함.

### 3) 시스템 패키지

```bash
apt-get install -y ffmpeg        # torchaudio 가 오디오를 읽을 때 부름
```

### 설치 뒤 확인

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 2.7.1+cu128 True          <- 이렇게 나와야 함
python -c "import wespeaker; print('wespeaker OK')"
ffmpeg -version | head -1
```

> **미검증** — 이 절차를 빈 서버에서 처음부터 돌려 본 적은 없음.
> 지금 환경은 여러 번 고쳐 가며 만든 것이고, 위 두 파일은 그 결과를 떠낸 것임.

### 이제 매번 할 일

```bash
conda activate wesep2
cd /workspace/git_clone/SD-FiLM/wesep/examples/librimix/tse/v2
```

**작업 디렉토리가 중요합니다.** `run.sh` 가 `path.sh` 를 읽어 `PYTHONPATH` 를 잡으므로,
다른 곳에서 부르면 `import wesep` 가 실패합니다.

---

## 1단계 — 데이터 준비 (stage 1~2, 한 번만)

이미 해 뒀다면 건너뜁니다. 확인하는 법:

```bash
ls data/clean/train-100/shard.list    # 이 파일이 있으면 끝난 것
```

없으면:

```bash
bash run.sh --stage 1 --stop-stage 2
```

| stage | 하는 일 | 걸리는 시간 |
|---|---|---|
| 1 | `/workspace/DB/Libri2Mix` 를 읽어 `wav.scp` · `utt2spk` · `spk2enroll.json` 생성 | 몇 분 |
| 2 | 그것을 **shard tar** 로 묶음 (`train-100` · `dev` · `test`) | 20~30분 |

**GPU 를 안 씁니다.** 데이터 경로는 [run.sh:17](run.sh#L17) 의 `Libri2Mix_dir` 가 가리킵니다.

---

## 2단계 — 학습과 평가 (stage 3~6, fusion 마다)

**실험 하나마다 stage 3~6 을 실행합니다.**
아래는 `FiLM` 을 예로 든 것이고, `concat` · `additive` · `multiply` 도
**config 와 `exp_dir` 이름만 바꾸면** 명령이 똑같습니다.

### 2-1. 학습 (stage 3)

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_FiLM.yaml \
  --exp_dir exp/bsrnn_ecapa_FiLM
```

| 인자 | 뜻 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | **물리 GPU 번호.** 이 프로세스는 그 한 장만 보게 됨 |
| `--config` | fusion 을 정하는 파일 |
| `--exp_dir` | 체크포인트와 로그가 쌓이는 곳. **fusion 마다 달라야 함** |

`--data` 는 주지 않습니다 — [run.sh:74](run.sh#L74) 이 기본값 `data` 에
`noise_type`(`clean`)을 붙여 `data/clean` 을 만듭니다.
`--data data/clean` 을 주면 `data/clean/clean` 이 되어 파일을 못 찾습니다.

**약 40시간**(150 epoch × 16분)입니다. 진행은 이렇게 봅니다:

```bash
tail -f exp/bsrnn_ecapa_FiLM/train.log
```

> **첫 스텝에서 약 113초 멈춘 것처럼 보입니다.** `compile_model: true` 라
> PyTorch 가 모델을 컴파일하는 시간입니다. 고장이 아닙니다.

#### 2-1-1. 중간에 끊겼다면

**같은 명령을 그대로 다시 치면 됩니다.** stage 3 이 `latest_checkpoint.pt` 를 자동으로
찾아 그 다음 epoch 부터 이어갑니다.

```bash
ls -la exp/bsrnn_ecapa_FiLM/models/latest_checkpoint.pt   # 어디까지 갔나
tail -3 exp/bsrnn_ecapa_FiLM/train.log
```

### 2-2. 평균 · 추론 · 채점 (stage 4~6)

학습이 끝나면 한 번에 돌립니다. **평균할 체크포인트를 고르는 방식이 두 가지**입니다.

**① 내가 고른 epoch 으로 — `best`**

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 4 --stop-stage 6 \
  --config confs/bsrnn_ecapa_FiLM.yaml \
  --exp_dir exp/bsrnn_ecapa_FiLM \
  --avg_mode best --avg_epochs "138,141"
```

**② 마지막 몇 개로 — `final`**

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 4 --stop-stage 6 \
  --config confs/bsrnn_ecapa_FiLM.yaml \
  --exp_dir exp/bsrnn_ecapa_FiLM \
  --avg_mode final --num_avg 2
```

**모드마다 따라오는 인자가 다릅니다** — `best` 면 `--avg_epochs`, `final` 이면 `--num_avg`.

### Table 2 만 필요하면 채점을 줄일 것

논문 Table 2 는 **SI-SDR 만** 보고합니다. PESQ · DNSMOS 는 쓰이지 않는데
stage 6 에서 시간을 많이 먹습니다 — 특히 DNSMOS 는 별도 딥러닝 모델을 돌립니다.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 4 --stop-stage 6 \
  --config confs/bsrnn_ecapa_FiLM.yaml \
  --exp_dir exp/bsrnn_ecapa_FiLM \
  --avg_mode best --avg_epochs "138,141" \
  --use_pesq false --use_dnsmos false
```

**stage 5 는 그대로 둬야 합니다** — SI-SDR 원값이 거기서 나옵니다.

| stage | 하는 일 | 시간 |
|---|---|---|
| 4 | 체크포인트 여러 개를 평균해 `avg_best_model.pt` 생성 | 1분 |
| 5 | test 셋 추론 — 분리 음원과 발화별 점수 | 25분 |
| 6 | 채점 — SI-SNRi · PESQ · STOI · DNSMOS | 10분 |

`--avg_epochs` 는 **검증 손실이 가장 낮은 epoch** 을 고르는 것입니다. 이렇게 찾습니다:

```bash
grep "Val info" exp/bsrnn_ecapa_FiLM/train.log | sort -t' ' -k7 -n | head -5
```

| `--avg_mode` | 무엇을 평균하나 | **쓰는 인자** | **무시하는 인자** |
|---|---|---|---|
| **`best`** | `--avg_epochs` 에 **내가 적은 epoch** | **`--avg_epochs`** | `--num_avg` |
| `final` | **마지막에서 `--num_avg` 개** | **`--num_avg`** | `--avg_epochs` |

두 모드는 **서로 다른 인자 하나씩만** 봅니다. 그래서 둘을 같이 줘도 겹치지 않습니다.

`best` 에서 `--num_avg` 를 안 줘도 되는 이유는 [run.sh:128](run.sh#L128) 가
`avg_epochs` 의 개수를 세어 자동으로 채우기 때문입니다 —
`"138,141"` 이면 2, `"135,140,145"` 면 3. 두 값이 어긋나
[average_model.py:83](../../../../wesep/bin/average_model.py#L83) 의 `assert` 에서 죽는 일을 막으려는 것입니다.

---

## 네 칸을 GPU 4장에 동시에 — **터미널 4개**

**터미널을 4개 열어 하나씩 띄웁니다.** 한 터미널에서 `&` 로 넷을 돌리면
진행 막대 네 개가 같은 줄을 써서 화면이 뒤섞이고, 어느 run 의 로그인지 구분이 안 됩니다.

각 터미널에서 먼저 환경과 위치를 잡습니다:

```bash
conda activate wesep2
cd /workspace/git_clone/SD-FiLM/wesep/examples/librimix/tse/v2
```

그다음 **자기 몫 한 줄**만 칩니다.

**터미널 1 — GPU 0 · concat**

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_concat.yaml --exp_dir exp/bsrnn_ecapa_concat
```

**터미널 2 — GPU 1 · additive**

```bash
CUDA_VISIBLE_DEVICES=1 bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_additive.yaml --exp_dir exp/bsrnn_ecapa_additive
```

**터미널 3 — GPU 2 · multiply**

```bash
CUDA_VISIBLE_DEVICES=2 bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_multiply.yaml --exp_dir exp/bsrnn_ecapa_multiply
```

**터미널 4 — GPU 3 · FiLM**

```bash
CUDA_VISIBLE_DEVICES=3 bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_FiLM.yaml --exp_dir exp/bsrnn_ecapa_FiLM
```

바뀌는 것은 **`CUDA_VISIBLE_DEVICES` · `--config` · `--exp_dir` 세 가지**뿐입니다.
**`exp_dir` 이 서로 달라야 합니다** — 같으면 체크포인트가 섞여 엉뚱한 가중치에서 학습이 이어집니다.

### 터미널을 4개 못 열 때

`tmux` 로 창을 나누거나, 로그를 파일로 보내고 나중에 봅니다:

```bash
CUDA_VISIBLE_DEVICES=0 nohup bash run.sh --stage 3 --stop-stage 3 \
  --config confs/bsrnn_ecapa_concat.yaml --exp_dir exp/bsrnn_ecapa_concat \
  > run_concat.out 2>&1 &
```

`exp/<이름>/train.log` 에도 같은 내용이 쌓이므로 **진행은 그 파일로 봐도 됩니다**:

```bash
tail -f exp/bsrnn_ecapa_concat/train.log
```

## 짧게 시험해 보기 — debug 모드

"제대로 도는가 · GPU 메모리가 모자라지 않는가" 만 30분 안에 확인하는 방법입니다.
**3 epoch** 만 돌고 평가도 데이터 일부만 씁니다.

```bash
CUDA_VISIBLE_DEVICES=0 bash run.sh --stage 3 --stop-stage 6 \
  --config confs/bsrnn_ecapa_FiLM.yaml \
  --exp_dir exp/bsrnn_ecapa_FiLM \
  --debug true
```

`--debug true` 를 주면 [confs/debug.yaml](confs/debug.yaml) 의 값이 본 config 위에 덮어씌워지고,
**결과 폴더가 `exp/bsrnn_ecapa_FiLM_debug` 로 자동으로 갈립니다.**
본 학습이 3 epoch 짜리 가중치를 이어받는 사고를 막으려는 것입니다.

---

## 결과가 어디에 쌓이나

```
exp/bsrnn_ecapa_FiLM/
├── train.log                      학습 로그 — "Val info val_loss" 로 수렴 확인
├── config.yaml                    이 run 이 실제로 쓴 설정 (자동 저장)
├── models/
│   ├── checkpoint_<N>.pt          epoch 별 체크포인트 (마지막 20개 보존)
│   ├── latest_checkpoint.pt  ->   가장 최근 것 (재개할 때 자동으로 읽음)
│   └── avg_best_model.pt          stage 4 가 만든 평균 모델
├── infer_utt_scores.csv           발화별 SI-SNR·SI-SNRi 원값
└── scoring/                       PESQ · STOI · DNSMOS
```

**Table 2 에 쓸 숫자는 `infer_utt_scores.csv` 에 있습니다** — stage 6 의 `scoring/` 에는
SI-SNRi 가 없어서, 이 파일이 유일한 출처입니다.

---

## 자주 걸리는 것

| 증상 | 원인과 해결 |
|---|---|
| `import wesep` 가 안 됨 | **작업 디렉토리가 틀림.** `examples/librimix/tse/v2` 에서 `run.sh` 를 불러야 `path.sh` 가 `PYTHONPATH` 를 잡음 |
| 로그의 GPU 번호가 항상 `0` | 정상임. `CUDA_VISIBLE_DEVICES=N` 을 주면 그 프로세스는 한 장만 보고 그것을 `0` 이라 부름 |
| 첫 스텝에서 2분 멈춤 | `compile_model: true` 의 컴파일 시간(약 113초). run 당 한 번뿐임 |
| 학습이 엉뚱한 가중치에서 시작 | **`exp_dir` 이 겹쳤음.** stage 3 은 그 폴더의 `latest_checkpoint.pt` 를 무조건 이어받음 |
| 체크포인트 로드에서 에러 | `strict=True` 라 키가 안 맞으면 죽음. 예전에는 조용히 넘어가 **랜덤 초기화로 학습되는 사고**가 났었음 |
| `--avg_epochs` 를 뭘 넣을지 모름 | `run.sh` 기본값은 `"138,141"` 임. **run 마다 `val_loss` 를 보고 다시 고를 것** |

---

## 이 레시피가 원본과 다른 점

wesep 원본에서 무엇을 왜 고쳤는지는 SD-FiLM 저장소의
`docs/issues/wesep_fusion_reproduction.md` 의 **원본 대비 수정 목록** 절에 표로 있습니다.
커밋 10개와 1:1로 대응합니다.
