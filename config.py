# config.py
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from argparse import ArgumentParser, Namespace
from typing import Any, Dict
import yaml
import sys

def _str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "on"):
        return True
    if s in ("false", "0", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot interpret boolean value from: {v}")

@dataclass
class Config:
    # === defaults (현재 코드와 동일) ===
    data: str = "./data/Smoothed_CyberTrend_Forecasting_All.csv"
    log_interval: int = 2000
    save: str = "model/Bayesian/model.safetensors"
    optim: str = "adam"
    L1Loss: bool = True
    normalize: int = 2
    device: str = "cuda:1"

    gcn_true: bool = True
    buildA_true: bool = True
    gcn_depth: int = 2
    num_nodes: int = 142
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    dilation_exponential: int = 2
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    in_dim: int = 1
    seq_in_len: int = 10
    seq_out_len: int = 36
    horizon: int = 1
    layers: int = 5

    batch_size: int = 8
    lr: float = 0.001
    weight_decay: float = 0.00001
    clip: int = 10

    propalpha: float = 0.05
    tanhalpha: float = 3

    epochs: int = 200
    num_split: int = 1
    step_size: int = 100

    search_iters: int = 60 # 60
    train_ratio: float = 0.43
    valid_ratio: float = 0.30
    hp_path: str = "model/Bayesian/hp.txt"

def _coerce_type(name: str, value: Any, cfg: Config):
    # cfg 필드 타입 기준으로 문자열을 적절히 형변환(특히 CLI override 시)
    t = {f.name: f.type for f in fields(cfg)}.get(name)
    if t is None or isinstance(value, t):
        return value
    try:
        if t is bool:
            return _str2bool(value)
        return t(value)
    except Exception:
        # 변환 실패 시 원래 값 유지
        return value

def _merge_dicts(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in extra.items():
        if v is None:
            continue
        out[k] = v
    return out

def load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("config yaml must contain a mapping at top-level.")
    return data

def get_args(argv: list[str] | None = None) -> Namespace:
    """
    1) 기본 Config
    2) --config yaml 로드
    3) 추가로 전달된 모든 CLI 키=값 override (예: --lr 0.0005, --device cuda:0)
       -> 존재하는 필드만 반영, 타입 자동 변환
    결과를 argparse.Namespace로 반환해서 기존 args.* 그대로 사용 가능.
    """
    # 1) 최소 파서: --config만 먼저 읽음(나머지는 unknown으로 두고 나중에 수동 머지)
    parser = ArgumentParser(add_help=True)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file")
    # unknown은 그대로 둔 다음 우리가 수동으로 병합/형변환
    known, unknown = parser.parse_known_args(args=argv)

    # 2) 기본값 → YAML → CLI override 순으로 병합
    cfg = Config()  # defaults
    yaml_dict = load_yaml(known.config)

    # CLI override 파싱: '--key value' 또는 '--flag=value' 스타일 모두 허용
    cli_kv: Dict[str, Any] = {}
    i = 0
    while i < len(unknown):
        tok = unknown[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok.lstrip("-")
        val: Any = True  # bool flag 스타일일 경우를 대비
        # --key=value
        if "=" in key:
            key, val = key.split("=", 1)
        else:
            # --key value
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                val = unknown[i + 1]
                i += 1
        cli_kv[key] = val
        i += 1

    # YAML 병합 (존재하는 필드만)
    merged = asdict(cfg)
    for k, v in yaml_dict.items():
        if k in merged:
            merged[k] = v

    # CLI 병합 (존재하는 필드만 + 타입 맞춤)
    for k, v in cli_kv.items():
        if k in merged:
            merged[k] = _coerce_type(k, v, cfg)

    # 최종 Config 생성
    cfg_final = Config(**merged)

    # argparse.Namespace로 반환 (기존 args.* 사용 호환)
    return Namespace(**asdict(cfg_final))


if __name__ == "__main__":
    # 디버그 출력용
    ns = get_args(sys.argv[1:])
    for k, v in vars(ns).items():
        print(f"{k}: {v!r}")
