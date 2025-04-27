# run_experiments.py

import shutil
import subprocess
import json
import re
import itertools
from pathlib import Path

# 1) src/config.py 경로 및 백업
CONFIG_PATH = Path('src/config.py')
BACKUP_PATH = Path('src/config.py.bak')
if not BACKUP_PATH.exists():
    shutil.copy(CONFIG_PATH, BACKUP_PATH)
orig_text = CONFIG_PATH.read_text()

# 2) 실험 파라미터 후보 목록
models = [
    'resnet18',
    'resnet34',
    'efficientnet_b0',
]

optimizers = [
    {'type': 'AdamW', 'lr': 5e-4,  'weight_decay': 1e-3},
]

schedulers = [
    {'type': 'MultiStepLR',       'milestones': [30, 35], 'gamma': 0.2},
    {'type': 'CosineAnnealingLR', 'T_max': 40},
]

batch_sizes = [256, 512]
num_epochs  = [40]

# 3) 모든 조합 생성
experiments = list(itertools.product(
    models, optimizers, schedulers, batch_sizes, num_epochs
))

# 4) 순차 실행
for idx, (model, opt, sched, bs, epochs) in enumerate(experiments, start=1):
    print(f'\n=== Experiment {idx}/{len(experiments)} ===')
    print(f' Model:     {model}')
    print(f' Optimizer: {opt}')
    print(f' Scheduler: {sched}')
    print(f' Batch sz:  {bs}')
    print(f' Epochs:    {epochs}')

    # 4-1) src/config.py 수정
    txt = orig_text
    txt = re.sub(
        r"MODEL_NAME\s*=\s*['\"].*?['\"]",
        f"MODEL_NAME = '{model}'",
        txt
    )
    txt = re.sub(
        r"OPTIMIZER_PARAMS\s*=\s*\{[^}]*\}",
        f"OPTIMIZER_PARAMS = {json.dumps(opt)}",
        txt,
        flags=re.DOTALL
    )
    txt = re.sub(
        r"SCHEDULER_PARAMS\s*=\s*\{[^}]*\}",
        f"SCHEDULER_PARAMS = {json.dumps(sched)}",
        txt,
        flags=re.DOTALL
    )
    txt = re.sub(
        r"BATCH_SIZE\s*=\s*\d+",
        f"BATCH_SIZE = {bs}",
        txt
    )
    txt = re.sub(
        r"NUM_EPOCHS\s*=\s*\d+",
        f"NUM_EPOCHS = {epochs}",
        txt
    )
    CONFIG_PATH.write_text(txt)

    # 4-2) train.py 실행 (프로젝트 루트에서 호출)
    subprocess.run(['python', 'train.py'], check=True)

# 5) 원본 복구
shutil.move(BACKUP_PATH, CONFIG_PATH)
print('\n✅ 모든 실험 완료, src/config.py 원본이 복구되었습니다.')
