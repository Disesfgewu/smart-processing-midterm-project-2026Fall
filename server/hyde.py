"""HyDE 提示擴展（介面預留，尚未實作）。

規劃：LLM 將原始提示改寫成數個 ≤12 字的外觀描述（OWL 文字上限 16 token），
全部作為查詢；同一物件取最高分描述並歸併回原始概念。詳見 docs/experiments.md。
"""
from typing import Dict, List

from applog import logger


def expand_prompts(prompts: List[str]) -> Dict[str, List[str]]:
    """回傳 {原始提示: [描述1, 描述2, ...]}。目前尚未實作，丟出 NotImplementedError。"""
    logger.warning(f"[hyde.expand_prompts]: input=(prompts={prompts}), output=not implemented")
    raise NotImplementedError("HyDE 尚未實作")
