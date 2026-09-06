from __future__ import annotations

import csv
import html
import itertools
import json
import math
import re
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go


APP_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = APP_ROOT / "vendor"
if str(VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDOR_ROOT))

from pathmat.agent_contracts import default_agent_contracts
from pathmat.agents import DeviceLimits, device_gate, protocol_gate
from pathmat.central_brain import BrainAction, BrainTask, CentralBrainAgent
from pathmat.central_brain_v2 import CentralBrainAgentV2
from pathmat.core import EvidenceLevel, EvidencePackage, critic_check, parse_formula
from pathmat.ec_pathwm_agent_adapter import ECPathWMAdapter
from pathmat.memory import MemoryStore


MODEL_PATH = APP_ROOT / "models" / "ec_pathwm_evidence_conditioned_v2.joblib"
REPORT_DIR = APP_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

_MODEL: ECPathWMAdapter | None = None


AGENT_ZH = {
    "Scout Agent": ("资料搜索", "寻找相关资料并列出缺少的信息"),
    "License-Provenance Agent": ("来源检查", "确认资料来自哪里以及是否可以使用"),
    "Curator Agent": ("数据整理", "统一记录格式并清理重复内容"),
    "Extraction Agent": ("信息提取", "读取配方、工艺步骤和实验条件"),
    "Physics Agent": ("条件检查", "检查化学式、制备路线和设备条件是否合理"),
    "Hypothesis Agent": ("多种解释", "列出目标产物、其他产物和资料不足等可能情况"),
    "Synthesis Planner Agent": ("路线建议", "整理下一步可以比较的路线"),
    "Characterization Agent": ("表征选择", "选择最能区分不同结果的检测方法"),
    "Adversarial Critic Agent": ("漏洞检查", "寻找结论中的证据缺口、反例和遗漏"),
    "Statistician Agent": ("可信度检查", "检查数值稳定性和资料覆盖程度"),
    "Safety Agent": ("执行限制检查", "检查设备条件、输入完整性和操作范围"),
    "Orchestrator Agent": ("任务协调", "安排各模块顺序并保存完整运行记录"),
}


PRESETS: dict[str, dict[str, Any]] = {
    "SrFe₁₂O₁₉｜覆盖域内功能演示": {
        "question": "这条 SrFe12O19 固相制备路线更接近目标产物还是其他产物？下一步需要补充哪些实验数据？",
        "target": "SrFe12O19",
        "precursors": "Fe2O3, SrCO3",
        "source_id": "lee2025_impurities",
        "family": "oxide",
        "actions": "混合, 加热, 研磨, 加热",
        "temperature": 1250,
        "duration": 12,
        "atmosphere": "air",
        "modalities": ["XRD"],
        "license_status": "source-licensed",
        "evidence_linked": True,
        "protocol_complete": True,
        "device_allows": True,
        "challenge": "无",
        "note": "示例路线已经填好。运行后可查看资料比较、检查结果和下一步建议。",
    },
    "SrFe₁₂O₁₉｜历史资料不足示例": {
        "question": "现有资料能否支持这条 SrFe12O19 路线？如果资料不足，还需要补充什么？",
        "target": "SrFe12O19",
        "precursors": "Fe2O3, SrCO3",
        "source_id": "kononova2019",
        "family": "oxide",
        "actions": "混合, 加热, 混合, 成型, 加热",
        "temperature": 1000,
        "duration": 8,
        "atmosphere": "air",
        "modalities": [],
        "license_status": "source_license_required",
        "evidence_linked": True,
        "protocol_complete": True,
        "device_allows": True,
        "challenge": "无",
        "note": "这个案例使用另一组历史资料，用于观察资料覆盖不足时系统如何处理。",
    },
    "ZnO｜外部来源与缺模态挑战": {
        "question": "不同气氛和退火条件下的 ZnO 记录有什么差异？还需要补充哪些检测数据？",
        "target": "ZnO",
        "precursors": "Zn",
        "source_id": "mendeley_jvs4nr5b74_zno",
        "family": "oxide",
        "actions": "切换气氛, 加热",
        "temperature": 550,
        "duration": 2,
        "atmosphere": "O2",
        "modalities": ["PL", "XPS"],
        "license_status": "CC BY 4.0",
        "evidence_linked": True,
        "protocol_complete": True,
        "device_allows": True,
        "challenge": "缺少关键表征",
        "note": "这个案例使用模型没有见过的新来源，用于观察系统如何提示资料不足。",
    },
    "自定义路线": {
        "question": "这条材料制备路线目前有哪些依据？下一步需要补充哪些实验数据？",
        "target": "LiFePO4",
        "precursors": "Li2CO3, FePO4",
        "source_id": "custom_source",
        "family": "phosphate_or_phosphide",
        "actions": "混合, 加热",
        "temperature": 700,
        "duration": 10,
        "atmosphere": "inert",
        "modalities": ["XRD"],
        "license_status": "unknown",
        "evidence_linked": False,
        "protocol_complete": False,
        "device_allows": True,
        "challenge": "无",
        "note": "请填写自己的材料、工艺条件和资料来源。来源无法追踪时，系统会提示需要补充资料。",
    },
}


def get_model() -> ECPathWMAdapter:
    global _MODEL
    if _MODEL is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"EC-PathWM model not found: {MODEL_PATH}")
        _MODEL = ECPathWMAdapter.load(MODEL_PATH)
    return _MODEL


class PhysicalRequestPlanner:
    """Create a real controller-level physical-actuation request for gate testing."""

    def propose(self, task: BrainTask, world_state: Any, memory: MemoryStore, cycle: int) -> list[BrainAction]:
        if cycle > 0:
            return []
        return [
            BrainAction(
                name="physical_synthesis_request",
                authority_level="physical_actuation",
                payload={"target": task.target, "request": "start_experiment"},
                source_ids=task.source_ids,
                rationale="Challenge-mode request used to verify the physical-actuation gate.",
                requires_review=True,
            )
        ]


def load_preset(name: str) -> tuple[Any, ...]:
    item = PRESETS.get(name, PRESETS["自定义路线"])
    return (
        item["question"],
        item["target"],
        item["precursors"],
        item["source_id"],
        item["family"],
        item["actions"],
        item["temperature"],
        item["duration"],
        item["atmosphere"],
        item["modalities"],
        item["license_status"],
        item["evidence_linked"],
        item["protocol_complete"],
        item["device_allows"],
        item["challenge"],
        item["note"],
    )


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return _safe_list(loaded)
        except json.JSONDecodeError:
            pass
    return [part.strip() for part in re.split(r"[,;，；]", text) if part.strip()]


def _parse_uploaded_file(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        loaded = json.loads(file_path.read_text(encoding="utf-8-sig"))
        if isinstance(loaded, list):
            loaded = loaded[0] if loaded else {}
        if not isinstance(loaded, Mapping):
            raise ValueError("JSON 顶层必须是对象，或至少包含一条对象记录的数组。")
        return dict(loaded)
    if suffix == ".csv":
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row is None:
            raise ValueError("CSV 不包含数据行。")
        return dict(row)
    raise ValueError("仅支持 .json 或 .csv 文件。")


def _uploaded_override(uploaded: Mapping[str, Any]) -> dict[str, Any]:
    if not uploaded:
        return {}
    source = uploaded.get("source") if isinstance(uploaded.get("source"), Mapping) else {}
    actions = uploaded.get("actions")
    if isinstance(actions, list):
        action_names = [str(item.get("name") or "Unknown") if isinstance(item, Mapping) else str(item) for item in actions]
    else:
        action_names = _safe_list(actions or uploaded.get("process_actions"))
    return {
        "target": uploaded.get("target_formula") or uploaded.get("target"),
        "precursors": uploaded.get("precursor_formulas") or uploaded.get("precursors"),
        "source_id": source.get("source_id") or uploaded.get("source_id"),
        "family": uploaded.get("chemistry_family") or uploaded.get("family"),
        "actions": action_names,
        "temperature": uploaded.get("temperature_C") or uploaded.get("temperature"),
        "duration": uploaded.get("duration_h") or uploaded.get("duration"),
        "atmosphere": uploaded.get("atmosphere"),
    }


def _make_actions(names: list[str], temperature: float, duration: float, atmosphere: str) -> list[dict[str, Any]]:
    aliases = {
        "混合": "Mixing",
        "加热": "HeatingOperation",
        "烧结": "HeatingOperation",
        "研磨": "Grinding",
        "液相研磨": "LiquidGrinding",
        "成型": "ShapingOperation",
        "溶液混合": "SolutionMixing",
        "淬火": "QuenchingOperation",
        "干燥": "DryingOperation",
        "等待": "Wait",
        "切换气氛": "AtmosphereSwitch",
    }
    allowed = {
        "Mixing", "SolutionMixing", "LiquidGrinding", "HeatingOperation",
        "DryingOperation", "ShapingOperation", "QuenchingOperation", "Wait",
        "AtmosphereSwitch", "Grinding", "Unknown",
    }
    output: list[dict[str, Any]] = []
    for raw_name in names or ["Unknown"]:
        normalized = aliases.get(raw_name, raw_name)
        name = normalized if normalized in allowed else "Unknown"
        action: dict[str, Any] = {"name": name}
        if name in {"HeatingOperation", "DryingOperation", "Wait"}:
            action["temperature_C"] = float(temperature)
            action["duration_h"] = float(duration)
            action["atmosphere"] = atmosphere
        elif name == "AtmosphereSwitch":
            action["atmosphere"] = atmosphere
        output.append(action)
    return output


def _badge(text: str, state: str) -> str:
    safe = html.escape(text)
    return f'<span class="pm-badge pm-{state}">{safe}</span>'


def _status_html(status: str, headline: str, detail: str) -> str:
    color = "ok" if status == "completed" else "warn" if status in {"abstained", "review_required"} else "bad"
    return (
        f'<div class="pm-status pm-status-{color}">'
        f'<div class="pm-status-kicker">运行状态</div>'
        f'<div class="pm-status-title">{html.escape(headline)}</div>'
        f'<div class="pm-status-detail">{html.escape(detail)}</div>'
        f'</div>'
    )


def _gate_html(gates: Mapping[str, dict[str, Any]]) -> str:
    friendly = {
        "Evidence/Critic": "资料来源清楚，结论范围与现有资料相符",
        "EC-PathWM": "当前材料和来源在模型可处理的范围内",
        "Device": "温度、时间和气氛符合已填写的设备条件",
        "Protocol": "本次分析需要的检查项已经填写完整",
        "Controller/Safety": "系统将生成分析结果和下一步建议",
    }
    display_names = {
        "Evidence/Critic": "资料与结论",
        "EC-PathWM": "模型适用范围",
        "Device": "设备条件",
        "Protocol": "输入完整性",
        "Controller/Safety": "操作范围",
    }
    cards = []
    for label, gate in gates.items():
        allowed = bool(gate["allowed"])
        state = "ok" if allowed else "bad"
        reasons = gate.get("reasons") or []
        detail = friendly.get(label, "本项检查通过") if allowed else "；".join(str(item) for item in reasons)
        cards.append(
            f'<div class="pm-gate"><div>{_badge("通过" if allowed else "需处理", state)}</div>'
            f'<h4>{html.escape(display_names.get(label, label))}</h4><p>{html.escape(detail)}</p></div>'
        )
    return '<div class="pm-gate-grid">' + "".join(cards) + "</div>"


def _trace_rows(
    *,
    source_present: bool,
    license_ok: bool,
    record_ok: bool,
    device_allowed: bool,
    critic_allowed: bool,
    stats_ok: bool,
    protocol_allowed: bool,
    brain_status: str,
    prediction: Mapping[str, Any] | None,
) -> list[list[str]]:
    contracts = default_agent_contracts()
    rows: list[list[str]] = []
    status_map = {
        "Scout Agent": ("完成" if source_present else "等待证据", "已绑定来源" if source_present else "未提供可追溯来源"),
        "License-Provenance Agent": ("完成" if license_ok else "需处理", "许可证可识别" if license_ok else "许可证未知或受限"),
        "Curator Agent": ("完成" if record_ok else "需处理", "记录字段已标准化" if record_ok else "目标化学式或字段无效"),
        "Extraction Agent": ("完成" if record_ok else "需处理", "工艺序列和条件已结构化" if record_ok else "结构化失败"),
        "Physics Agent": ("完成" if device_allowed and record_ok else "需处理", "公式/设备条件通过" if device_allowed and record_ok else "存在公式或设备约束问题"),
        "Hypothesis Agent": ("完成" if prediction else "等待", "已列出目标产物、其他产物和资料不足三种解释" if prediction else "模型还没有产生可用结果"),
        "Synthesis Planner Agent": ("建议已生成" if prediction else "等待", "已整理可以继续比较的路线" if prediction else "等待资料"),
        "Characterization Agent": ("完成", "根据缺失模态生成验证建议"),
        "Adversarial Critic Agent": ("通过" if critic_allowed else "需处理", "没有发现足以推翻当前分析的资料问题" if critic_allowed else "来源、使用条件或验证方法存在问题"),
        "Statistician Agent": ("通过" if stats_ok else "需处理", "数值稳定性和资料覆盖程度在设定范围内" if stats_ok else "数值波动或资料覆盖仍需处理"),
        "Safety Agent": ("通过" if protocol_allowed and device_allowed and brain_status != "policy_blocked" else "阻止", "设备条件和输入检查通过" if protocol_allowed and device_allowed and brain_status != "policy_blocked" else "设备条件、输入内容或操作范围未通过"),
        "Orchestrator Agent": ("完成", f"全部模块运行状态：{brain_status}"),
    }
    for role, contract in contracts.items():
        zh, _ = AGENT_ZH[role]
        status, finding = status_map[role]
        rows.append([zh, status, finding])
    return rows


def _recommendation(
    modalities: list[str],
    prediction: Mapping[str, Any] | None,
    brain_status: str,
    gates: Mapping[str, dict[str, Any]],
) -> tuple[str, list[str], str, str]:
    if brain_status == "policy_blocked":
        return "当前生成实验方案和检查记录", ["请求中包含真实设备操作", "系统已保存分析记录，设备控制部分保持关闭"], "controller_policy_block", "—"
    if prediction and prediction.get("abstain"):
        reason_map = {
            "unseen_source": "系统没有见过这个资料来源",
            "unseen_family": "系统缺少这个材料类别的参考记录",
            "low_source_coverage": "这个来源中的可用记录太少",
            "high_epistemic_uncertainty": "不同模型给出的判断差异较大",
            "high_evidence_risk": "现有资料覆盖不足",
        }
        reasons = [reason_map.get(str(item), str(item)) for item in prediction.get("abstention_reasons", [])]
        return "当前资料不足，请先补充来源和相近案例", reasons or ["当前输入超出模型较有把握的范围"], "model_abstention", "—"
    failed_gates = [name for name, gate in gates.items() if not gate.get("allowed") and name != "EC-PathWM"]
    if failed_gates:
        return "先解决未通过的检查项，再解释模型结果", [f"待处理：{name}" for name in failed_gates], "scientific_gate_failure", "—"
    if "XRD" not in modalities:
        return "先补 XRD，确认产物到底是什么相", ["当前没有可用于相鉴定的 XRD 证据", "需要区分目标产物和其他可能产物（杂相）", "新结果必须绑定到同一样本和来源"], "missing_xrd", "—"

    competitor = max((prediction.get("competitor_probabilities") or {"none": 0.0}).values()) if prediction else 0.0
    path_score = float(prediction.get("path_score", 0.0)) if prediction else 0.0
    evidence_risk = float(prediction.get("evidence_risk", 1.0)) if prediction else 1.0
    next_actions = prediction.get("next_action_probability") or {} if prediction else {}
    if next_actions:
        top_action, top_probability = max(next_actions.items(), key=lambda item: float(item[1]))
        top_action_text = f"{top_action} ({float(top_probability):.1%})"
    else:
        top_action, top_probability, top_action_text = "", 0.0, "—"

    if competitor >= 0.40:
        missing_crosscheck = "Raman" not in modalities and "SEM" not in modalities
        reasons = [
            f"模型对其他可能产物的倾向达到 {competitor:.1%}",
            "对现有 XRD 做峰归属和定量相分析，并与候选杂相参考谱比较",
        ]
        if missing_crosscheck:
            reasons.append("补充 Raman 或 SEM，从另一个角度核对 XRD 结果")
        else:
            reasons.append("利用已有的其他表征检查 XRD 解释是否一致")
        return "先确认产物的物相组成，再讨论工艺优化", reasons, "high_competing_path_risk", top_action_text

    if path_score < 0.25:
        return (
            "这条路线优势太弱，先与备选路线做同标准比较",
            [f"当前路线的综合比较分数为 {path_score:.3f}", "使用同一套判断标准比较前驱体、工艺顺序或气氛方案"],
            "low_path_score",
            top_action_text,
        )
    if evidence_risk > 0.20:
        return (
            "模型对这类数据不够熟悉，先补同来源证据",
            [f"当前资料覆盖不足程度为 {evidence_risk:.1%}", "先增加同来源记录，再重新比较路线"],
            "evidence_risk",
            top_action_text,
        )
    if not ({"Raman", "SEM"} & set(modalities)):
        return (
            "增加一种独立表征，检查 XRD 结论是否稳定",
            ["目前已有 XRD，还需要另一种方法从不同角度核对", "可用 Raman 检查局域结构，或用 SEM 观察形貌和不均一性"],
            "missing_cross_modal_validation",
            top_action_text,
        )

    action_guidance = {
        "Mixing": "复核混合均匀性与前驱体比例的来源证据",
        "HeatingOperation": "设计一次只改变温度或时间的对照实验，确认条件变化与结果之间的关系",
        "Grinding": "比较研磨前后路径记录与粒度相关证据",
        "ShapingOperation": "检查成型步骤是否改变接触界面和后续相形成记录",
        "SolutionMixing": "比较溶液混合与固相混合的同标准历史记录",
        "QuenchingOperation": "核验淬火路径的历史来源与可逆计算证据",
        "DryingOperation": "检查干燥条件对前驱体状态的来源记录",
    }.get(str(top_action), "比较候选路径，并设计一次只改变一个条件的对照实验")
    return (
        action_guidance,
        [f"相似历史路线中最常见的后续步骤为 {top_action_text}", "可据此查找相似路线并设计对照"],
        "next_action_proxy",
        top_action_text,
    )


def _decision_markdown(
    question: str,
    target: str,
    source_id: str,
    precursors: list[str],
    action_names: list[str],
    temperature: float,
    duration: float,
    atmosphere: str,
    modalities: list[str],
    prediction: Mapping[str, Any] | None,
    recommendation: str,
    recommendation_reasons: list[str],
    recommendation_trigger: str,
    top_next_action: str,
    gates: Mapping[str, dict[str, Any]],
    brain_status: str,
) -> str:
    action_zh = {
        "Mixing": "混合",
        "HeatingOperation": "加热/烧结",
        "Grinding": "研磨",
        "LiquidGrinding": "液相研磨",
        "ShapingOperation": "成型",
        "SolutionMixing": "溶液混合",
        "QuenchingOperation": "淬火",
        "DryingOperation": "干燥",
        "AtmosphereSwitch": "切换气氛",
    }
    route_text = " → ".join(action_zh.get(item, item) for item in action_names) or "未提供"
    precursor_text = " + ".join(precursors) or "未提供"
    modality_text = "、".join(modalities) if modalities else "暂无关键表征"
    passed_gates = sum(1 for gate in gates.values() if gate["allowed"])
    total_gates = len(gates)

    if prediction:
        target_value = float(prediction.get("target_probability_proxy", 0.0))
        competitor = max((prediction.get("competitor_probabilities") or {"none": 0.0}).values())
        evidence_risk_value = float(prediction.get("evidence_risk", 0.0))
        score_value = float(prediction.get("path_score", 0.0))
        target_proxy = f"{100 * target_value:.1f}%"
        competitor_proxy = f"{100 * float(competitor):.1f}%"
        margin = 100 * (target_value - float(competitor))
        margin_text = f"{margin:+.1f} 个百分点"
        familiarity = 100 * (1.0 - evidence_risk_value)
        familiarity_text = f"{familiarity:.1f}%"
        risk = f"{100 * evidence_risk_value:.1f}%"
        path_score = f"{score_value:.3f}"

        if margin >= 20:
            route_plain = "根据现有资料，模型明显偏向目标产物"
        elif margin >= 10:
            route_plain = "根据现有资料，模型较偏向目标产物"
        elif margin >= 0:
            route_plain = "模型对目标产物和其他可能产物的判断十分接近"
        else:
            route_plain = "模型更偏向其他可能产物"

        if evidence_risk_value <= 0.10:
            evidence_plain = "模型对当前来源和材料类别较熟悉"
        elif evidence_risk_value <= 0.35:
            evidence_plain = "模型有一定参考记录，但覆盖仍不充分"
        else:
            evidence_plain = "模型对当前来源或材料类别不熟悉"
    else:
        target_proxy = competitor_proxy = risk = path_score = "—"
        margin_text = familiarity_text = "—"
        route_plain = "没有得到可解释的模型结果"
        evidence_plain = "无法判断数据覆盖情况"

    if brain_status == "policy_blocked":
        verdict = "🔴 当前请求包含设备操作，系统已停止执行部分"
        can_conclude = "当前请求包含设备执行操作，系统保留了分析部分并停止执行部分。"
    elif brain_status == "abstained" or (prediction and prediction.get("abstain")):
        verdict = "🔴 当前资料不足，请先补充信息"
        can_conclude = "当前资料还不支持结论，需要先补充来源或相近材料记录。"
    elif passed_gates < total_gates:
        verdict = "🟠 分析已完成，还有检查项需要处理"
        can_conclude = "先处理没有通过的检查项，再判断路线。"
    elif prediction and abs(100 * (float(prediction.get("target_probability_proxy", 0.0)) - competitor)) < 10:
        verdict = "🟠 两种结果非常接近，需要实验进一步确认"
        can_conclude = "目标产物与其他可能产物十分接近，需要实验表征进一步区分。"
    else:
        verdict = "🟡 当前路线可以进入实验验证阶段"
        can_conclude = "当前结果适合进入实验验证阶段，实际结果由测量和重复实验确定。"

    bullets = "\n".join(f"- {item}" for item in recommendation_reasons)
    status_zh = {
        "completed": "已完成",
        "abstained": "等待补充资料",
        "policy_blocked": "已停止设备操作",
        "external_dependency": "等待外部条件",
    }.get(brain_status, brain_status)
    return f"""## 分析结果

### {verdict}

**一句话解释：**{route_plain}。下一步用实验表征区分**目标产物**和**其他可能产物（杂相）**。

> 页面里的百分比用于比较模型对两种结果的倾向。实际结果由实验测量和重复验证确定。

### 本次到底评估了什么

- **目标产物：** `{target}`
- **前驱体：** `{precursor_text}`
- **工艺路线：** {route_text}
- **主要条件：** {temperature:g} °C，{duration:g} h，{atmosphere}
- **已有表征：** {modality_text}
- **资料来源：** `{source_id or '未填写'}`

### 四个问题，直接看答案

1. **当前更像目标相还是杂相？**  
   {route_plain}：模型对目标产物的倾向为 **{target_proxy}**，对其他可能产物的倾向为 **{competitor_proxy}**，两者相差 **{margin_text}**。

2. **现有资料够不够模型参考？**  
   {evidence_plain}。模型参考资料的覆盖程度约为 **{familiarity_text}**；数值越高，说明可供比较的相近记录越多。

3. **当前能把它当成可靠路线吗？**  
   **{can_conclude}** 当前记录中还缺少真实相含量、纯度和重复实验结果。

4. **软件有没有正常跑完？**  
   分析流程状态为 **{status_zh}**，结果检查通过 **{passed_gates}/{total_gates}**。实验进度需要根据实际样本记录确认。

### 接下来怎么做：{recommendation}

{bullets}

### 系统如何分工

- **资料搜索模块：** 找到目标产物和候选杂相的来源及参考图谱。
- **路线判断模块：** 比较现有资料更接近哪一种结果。
- **表征选择模块：** 选择最能区分不同结果的 XRD、Raman 或 SEM 数据。
- **漏洞与可信度检查：** 查找资料缺失、覆盖不足和数值波动带来的影响。
- **任务协调模块：** 汇总各项结果并生成下一步建议。

### 详细数值

| 技术字段 | 数值 | 应该怎么读 |
|---|---:|---|
| 模型对目标产物的倾向 | **{target_proxy}** | 与其他可能产物进行相对比较 |
| 模型对其他可能产物的倾向 | **{competitor_proxy}** | 两项越接近，判断越需要实验确认 |
| 两类支持度差距 | **{margin_text}** | 当前差距越小，路线判断越不稳 |
| 资料覆盖不足程度 | **{risk}** | 表示相近来源和材料记录的缺少程度 |
| 综合路线比较分数 | **{path_score}** | 用于排列候选路线的先后顺序 |

> 适用范围：帮助选择下一步需要补充的数据。材料的相含量、纯度和稳定性以实际测量结果为准。
"""


def _hypothesis_markdown(
    target: str,
    modalities: list[str],
    prediction: Mapping[str, Any] | None,
    recommendation_trigger: str,
) -> str:
    competitor = 0.0
    if prediction:
        competitor = max((prediction.get("competitor_probabilities") or {"none": 0.0}).values())
    missing = [item for item in ["XRD", "Raman", "SEM"] if item not in modalities]
    xrd_step = (
        "对同一样本采集 XRD，并完成目标相与候选杂相的峰归属。"
        if "XRD" not in modalities
        else "在已有 XRD 数据上继续完成峰归属、候选杂相比对和定量相分析。"
    )
    return f"""## 下一步怎么把结果变成可验证结论

### 当前需要区分三种可能结果

1. **形成目标产物：** 当前路线可能形成了 `{target}`，现有资料给出了这一方向的初步信号。
2. **同时形成其他产物：** 产物中可能含有杂相；模型对这一结果的倾向为 **{competitor:.1%}**。
3. **现有数据还不够：** 两项数值接近，也可能来自缺少能够区分两种结果的表征。

### 建议按这个顺序补证据

1. **先回答“产物是什么相”—— XRD**  
   {xrd_step}负责部分：**表征选择 + 其他产物比对**。

2. **再回答“XRD 的解释是否唯一”—— Raman 或 SEM**  
   对同一样本做一种独立交叉验证；Raman用于补充局域结构信息，SEM用于观察形貌和不均一性。缺失模态：**{', '.join(missing) if missing else '无'}**。

3. **再检查工艺条件与结果的关系—— 单变量对照实验**
   每次只改变温度、时间、气氛或前驱体中的一个因素，其余条件保持一致，避免把多因素变化误认为单一机制。

4. **把结果重新交回系统**  
   新证据必须保留同一样本编号、来源和工艺记录，再次运行模型，观察目标相与竞争相的差距是否稳定扩大。

### 什么结果才算支持或否定当前路线

- **支持目标路线：** 目标相特征峰清楚、候选杂相下降，并且独立表征给出一致解释。
- **否定当前路线：** 候选杂相在重复样本中稳定出现，或改变单一工艺因素后竞争相仍占主导。
- **仍然无法判断：** 表征之间互相矛盾、样本无法对应、来源缺失，或系统认为资料不足。

系统按照当前资料缺口和结果差距安排上述验证顺序。
"""


def _plot_layout(title: str, subtitle: str) -> dict[str, Any]:
    return {
        "title": {
            "text": f"{title}<br><sup>{subtitle}</sup>",
            "x": 0.02,
            "font": {"size": 21, "color": "#f2f7fb"},
        },
        "paper_bgcolor": "#07111f",
        "plot_bgcolor": "#07111f",
        "font": {"family": "Inter, Segoe UI, sans-serif", "color": "#c9d8e6"},
        "margin": {"l": 12, "r": 12, "t": 82, "b": 12},
        "height": 610,
        "showlegend": False,
    }


def _empty_3d_figure(title: str, subtitle: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(**_plot_layout(title, subtitle))
    figure.add_annotation(
        text="运行分析后生成交互视图",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 18, "color": "#8ba1b6"},
    )
    return figure


def _route_landscape_figure(prediction: Mapping[str, Any] | None) -> go.Figure:
    """Visualize how the implemented route score changes with two input indicators."""
    if not prediction:
        return _empty_3d_figure("3D 路线判断地图", "运行分析后显示当前路线的位置")

    target = float(prediction.get("target_probability_proxy", 0.5))
    risk = float(prediction.get("evidence_risk", 1.0))
    uncertainty = float(prediction.get("uncertainty_aleatoric_proxy", 0.0)) + float(
        prediction.get("uncertainty_epistemic_proxy", 0.0)
    )
    current_score = float(prediction.get("path_score", 0.0))

    target_axis = np.linspace(0.05, 0.95, 61)
    risk_axis = np.linspace(0.0, 0.60, 45)
    xx, yy = np.meshgrid(target_axis, risk_axis)
    zz = np.log(xx) - np.log(1.0 - xx) - uncertainty - yy

    figure = go.Figure()
    figure.add_trace(
        go.Surface(
            x=100 * xx,
            y=100 * yy,
            z=zz,
            colorscale=[
                [0.0, "#721c36"],
                [0.32, "#c34f36"],
                [0.50, "#d89a36"],
                [0.68, "#27a6a1"],
                [1.0, "#5de1c5"],
            ],
            opacity=0.88,
            colorbar={"title": "综合比较分数", "thickness": 13, "len": 0.64},
            contours={"z": {"show": True, "usecolormap": True, "highlightcolor": "#ffffff", "project_z": True}},
            hovertemplate="模型对目标产物的倾向 %{x:.1f}%<br>资料覆盖不足程度 %{y:.1f}%<br>综合比较分数 %{z:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[100 * target],
            y=[100 * risk],
            z=[current_score],
            mode="markers+text",
            text=["当前路线"],
            textposition="top center",
            marker={"size": 9, "color": "#ffffff", "line": {"color": "#ff6b3d", "width": 5}},
            hovertemplate=(
                f"当前路线<br>模型对目标产物的倾向 {100 * target:.1f}%<br>"
                f"资料覆盖不足程度 {100 * risk:.1f}%<br>综合比较分数 {current_score:.3f}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        **_plot_layout(
            "3D 路线判断地图",
            "拖动旋转 · 滚轮缩放 · 悬停查看数值｜观察两个输入指标如何影响综合比较分数",
        ),
        scene={
            "xaxis": {"title": "模型对目标产物的倾向 (%)", "gridcolor": "#24384c", "backgroundcolor": "#07111f"},
            "yaxis": {"title": "资料覆盖不足程度 (%)", "gridcolor": "#24384c", "backgroundcolor": "#07111f"},
            "zaxis": {"title": "综合比较分数", "gridcolor": "#24384c", "backgroundcolor": "#07111f"},
            "camera": {"eye": {"x": 1.55, "y": 1.55, "z": 1.0}},
            "aspectratio": {"x": 1.45, "y": 1.0, "z": 0.9},
        },
    )
    return figure


def _agent_network_figure(trace_rows: list[list[str]]) -> go.Figure:
    """Render the configured Agent/model workflow as an interactive 3D network."""
    if not trace_rows:
        return _empty_3d_figure("系统协作流程", "运行分析后显示各模块的连接和状态")

    positions = {
        "Scout Agent": (0.0, 2.4, 0.8),
        "License-Provenance Agent": (0.0, 0.8, -0.4),
        "Curator Agent": (0.0, -0.8, 0.4),
        "Extraction Agent": (0.0, -2.4, -0.8),
        "Physics Agent": (1.7, 2.0, -0.6),
        "EC-PathWM": (1.7, 0.65, 1.2),
        "Hypothesis Agent": (1.7, -0.65, 0.0),
        "Synthesis Planner Agent": (1.7, -2.0, -1.0),
        "Characterization Agent": (3.4, 2.0, 0.9),
        "Adversarial Critic Agent": (3.4, 0.65, -0.7),
        "Statistician Agent": (3.4, -0.65, 0.9),
        "Safety Agent": (3.4, -2.0, -0.7),
        "Orchestrator Agent": (5.1, 0.0, 0.25),
    }
    edges = [
        ("Scout Agent", "License-Provenance Agent"),
        ("Scout Agent", "Curator Agent"),
        ("License-Provenance Agent", "Extraction Agent"),
        ("Curator Agent", "Extraction Agent"),
        ("Extraction Agent", "Physics Agent"),
        ("Extraction Agent", "EC-PathWM"),
        ("Physics Agent", "EC-PathWM"),
        ("EC-PathWM", "Hypothesis Agent"),
        ("Hypothesis Agent", "Synthesis Planner Agent"),
        ("Hypothesis Agent", "Characterization Agent"),
        ("EC-PathWM", "Statistician Agent"),
        ("Characterization Agent", "Adversarial Critic Agent"),
        ("License-Provenance Agent", "Adversarial Critic Agent"),
        ("Adversarial Critic Agent", "Safety Agent"),
        ("Statistician Agent", "Safety Agent"),
        ("Synthesis Planner Agent", "Safety Agent"),
        ("Adversarial Critic Agent", "Orchestrator Agent"),
        ("Statistician Agent", "Orchestrator Agent"),
        ("Safety Agent", "Orchestrator Agent"),
    ]
    role_order = list(default_agent_contracts())
    row_by_name = {role: row for role, row in zip(role_order, trace_rows)}
    row_by_name["EC-PathWM"] = ["路线判断模型", "完成", "比较目标产物与其他可能产物，并检查资料是否充足"]

    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    edge_z: list[float | None] = []
    for start, end in edges:
        x0, y0, z0 = positions[start]
        x1, y1, z1 = positions[end]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_z.extend([z0, z1, None])

    figure = go.Figure()
    figure.add_trace(
        go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode="lines",
            line={"color": "rgba(80,189,207,.42)", "width": 4},
            hoverinfo="skip",
        )
    )

    names = list(positions)
    colors = []
    sizes = []
    symbols = []
    hover = []
    display = []
    for name in names:
        row = row_by_name.get(name, [name, "等待", "等待运行"])
        status = str(row[1])
        if any(token in status for token in ["阻止", "否决", "错误", "需处理"]):
            color = "#ff5f56"
        elif any(token in status for token in ["等待"]):
            color = "#f6b94a"
        elif name == "Orchestrator Agent":
            color = "#ffcc66"
        elif name == "EC-PathWM":
            color = "#63e6ff"
        else:
            color = "#51d6a9"
        colors.append(color)
        sizes.append(13 if name in {"EC-PathWM", "Orchestrator Agent"} else 9)
        symbols.append("diamond" if name == "EC-PathWM" else "square" if name == "Orchestrator Agent" else "circle")
        display.append(str(row[0]))
        hover.append(
            f"<b>{html.escape(str(row[0]))}</b><br>技术名称：{html.escape(name)}<br>状态：{html.escape(status)}"
            f"<br>{html.escape(str(row[2]))}<extra></extra>"
        )

    figure.add_trace(
        go.Scatter3d(
            x=[positions[name][0] for name in names],
            y=[positions[name][1] for name in names],
            z=[positions[name][2] for name in names],
            mode="markers+text",
            text=display,
            textposition="top center",
            textfont={"size": 10, "color": "#e8f2fa"},
            marker={"size": sizes, "color": colors, "symbol": symbols, "line": {"color": "#dffcff", "width": 1}},
            hovertemplate=hover,
        )
    )
    figure.add_trace(
        go.Scatter3d(
            x=[0.0, 1.7, 3.4, 5.1],
            y=[3.25, 3.25, 3.25, 3.25],
            z=[1.35, 1.35, 1.35, 1.35],
            mode="text",
            text=["资料准备", "路线分析", "结果检查", "汇总建议"],
            textfont={"size": 11, "color": "#65dce4"},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        **_plot_layout(
            "系统协作流程：12 个模块 + 路线判断模型",
            "拖动旋转 · 悬停查看每个部分负责什么、运行状态和本次发现",
        ),
        scene={
            "xaxis": {"visible": False, "showbackground": False},
            "yaxis": {"visible": False, "showbackground": False},
            "zaxis": {"visible": False, "showbackground": False},
            "bgcolor": "#07111f",
            "camera": {"eye": {"x": 1.65, "y": 1.4, "z": 0.85}},
            "aspectratio": {"x": 1.8, "y": 1.25, "z": 0.9},
        },
    )
    return figure


def _numeric_options(value: Any, label: str, minimum: float, maximum: float) -> list[float]:
    values: list[float] = []
    for item in _safe_list(value):
        try:
            number = float(item)
        except ValueError as exc:
            raise ValueError(f"{label}中包含无法识别的数值：{item}") from exc
        if number < minimum or number > maximum:
            raise ValueError(f"{label}需要在 {minimum:g} 到 {maximum:g} 之间：{number:g}")
        if number not in values:
            values.append(number)
    if not values:
        raise ValueError(f"请至少填写一个{label}")
    return values


def _screening_figure(
    baseline_runs: int,
    guided_runs: int,
    baseline_hours: float,
    guided_hours: float,
) -> go.Figure:
    run_ratio = 100.0 * guided_runs / baseline_runs if baseline_runs else 0.0
    hour_ratio = 100.0 * guided_hours / baseline_hours if baseline_hours else 0.0
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=[100.0, 100.0],
            y=["实验次数", "设备占用时间"],
            orientation="h",
            name="全部组合",
            marker_color="#52687c",
            text=[f"{baseline_runs} 次", f"{baseline_hours:.1f} h"],
            textposition="auto",
            hovertemplate="%{y}<br>%{text}<br>完整工作量的 100%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=[run_ratio, hour_ratio],
            y=["实验次数", "设备占用时间"],
            orientation="h",
            name="模型指导第一轮",
            marker_color="#45d2b0",
            text=[f"{guided_runs} 次", f"{guided_hours:.1f} h"],
            textposition="auto",
            hovertemplate="%{y}<br>%{text}<br>完整工作量的 %{x:.1f}%<extra></extra>",
        )
    )
    layout = _plot_layout(
        "模型指导的第一轮保留多少工作量",
        "灰色为全部组合；绿色为先做推荐条件。条形长度按完整工作量的百分比显示。",
    )
    layout.update(
        height=450,
        barmode="group",
        bargap=0.28,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.20, "xanchor": "left", "x": 0},
        xaxis={"range": [0, 108], "ticksuffix": "%", "gridcolor": "#21364a"},
        yaxis={"autorange": "reversed", "gridcolor": "#21364a"},
    )
    figure.update_layout(**layout)
    return figure


def _condition_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    temperature_span: float,
    duration_span: float,
) -> float:
    temperature_gap = abs(float(left["temperature"]) - float(right["temperature"])) / temperature_span
    duration_gap = abs(float(left["duration"]) - float(right["duration"])) / duration_span
    atmosphere_gap = float(str(left["atmosphere"]) != str(right["atmosphere"]))
    return float(math.sqrt((temperature_gap**2 + duration_gap**2 + atmosphere_gap**2) / 3.0))


def _choose_first_round(
    candidates: list[dict[str, Any]],
    count: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Select an informative, diverse first batch without inventing model separation."""
    usable = [item for item in candidates if not item["abstain"]]
    if not usable:
        return [], False

    score_values = [float(item["score"]) for item in usable]
    score_min, score_max = min(score_values), max(score_values)
    score_span = score_max - score_min
    model_separates_conditions = score_span > 1e-6
    weak_score_separation = score_span < 0.01
    temperatures = [float(item["temperature"]) for item in usable]
    durations = [float(item["duration"]) for item in usable]
    temperature_span = max(max(temperatures) - min(temperatures), 1.0)
    duration_span = max(max(durations) - min(durations), 1.0)
    max_hours = max(float(item["hours_per_run"]) for item in usable) or 1.0

    selected: list[dict[str, Any]] = []
    remaining = list(usable)
    while remaining and len(selected) < count:
        best_item: dict[str, Any] | None = None
        best_value = -math.inf
        for item in remaining:
            if selected:
                diversity = min(
                    _condition_distance(item, chosen, temperature_span, duration_span)
                    for chosen in selected
                )
            else:
                diversity = 0.0
            uncertainty = max(0.0, 1.0 - abs(float(item["target"]) - float(item["other"])))
            time_efficiency = max(0.0, 1.0 - float(item["hours_per_run"]) / max_hours)
            score_signal = (
                (float(item["score"]) - score_min) / score_span
                if model_separates_conditions
                else 0.5
            )
            if selected:
                value = (
                    0.18 * score_signal + 0.65 * diversity + 0.15 * uncertainty + 0.02 * time_efficiency
                    if model_separates_conditions and weak_score_separation
                    else 0.45 * score_signal + 0.35 * diversity + 0.15 * uncertainty + 0.05 * time_efficiency
                    if model_separates_conditions
                    else 0.58 * diversity + 0.27 * uncertainty + 0.15 * time_efficiency
                )
            else:
                value = (
                    0.68 * score_signal + 0.22 * uncertainty + 0.10 * time_efficiency
                    if model_separates_conditions
                    else 0.65 * uncertainty + 0.35 * time_efficiency
                )
            if value > best_value:
                best_item, best_value = item, value
        assert best_item is not None
        best_item["selection_value"] = float(best_value)
        selected.append(best_item)
        remaining.remove(best_item)

    return selected, model_separates_conditions


def _selection_reason(
    item: Mapping[str, Any],
    selected_before: list[Mapping[str, Any]],
    temperatures: list[float],
    durations: list[float],
    model_separates_conditions: bool,
) -> str:
    parts: list[str] = []
    if model_separates_conditions:
        parts.append("参考模型排序分")
    if not selected_before:
        parts.append("用较短时间取得首批结果")
    else:
        conditions: list[str] = []
        temperature = float(item["temperature"])
        duration = float(item["duration"])
        if temperature == min(temperatures):
            conditions.append("温度下限")
        elif temperature == max(temperatures):
            conditions.append("温度上限")
        if duration == min(durations):
            conditions.append("短时")
        elif duration == max(durations):
            conditions.append("长时")
        if str(item["atmosphere"]) not in {str(value["atmosphere"]) for value in selected_before}:
            conditions.append("新气氛")
        parts.append("覆盖" + "、".join(conditions or ["与已选组差异较大的条件"]))
    return "；".join(parts)


def _write_screening_plan_csv(
    *,
    target: str,
    precursors: list[str],
    source_id: str,
    actions: list[str],
    rows: list[list[Any]],
) -> str:
    plan_id = f"pathmat_experiment_plan_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    path = REPORT_DIR / f"{plan_id}.csv"
    fieldnames = [
        "plan_order",
        "first_round_status",
        "target_formula",
        "precursor_formulas",
        "source_id",
        "process_actions",
        "temperature_C",
        "duration_h",
        "atmosphere",
        "model_ranking_score",
        "position_in_batch",
        "selection_reason",
        "sample_id",
        "actual_phase_from_XRD",
        "Raman_or_SEM_note",
        "experiment_conclusion",
        "completed_at",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "plan_order": row[0],
                    "first_round_status": row[6],
                    "target_formula": target,
                    "precursor_formulas": ", ".join(precursors),
                    "source_id": source_id,
                    "process_actions": ", ".join(actions),
                    "temperature_C": row[1],
                    "duration_h": row[2],
                    "atmosphere": row[3],
                    "model_ranking_score": row[4],
                    "position_in_batch": row[5],
                    "selection_reason": row[7],
                    "sample_id": "",
                    "actual_phase_from_XRD": "",
                    "Raman_or_SEM_note": "",
                    "experiment_conclusion": "",
                    "completed_at": "",
                }
            )
    return str(path)


def screen_experiments(
    target: str,
    precursors: Any,
    source_id: str,
    family: str,
    actions: Any,
    modalities: Any,
    evidence_linked: bool,
    temperature_options: Any,
    duration_options: Any,
    atmosphere_options: Any,
    repeats: int,
    priority_count: int,
    characterization_hours: float,
) -> tuple[str, list[list[Any]], go.Figure, str | None]:
    """Rank a user-defined condition grid and estimate the resulting experiment workload."""
    try:
        target = str(target or "").strip()
        if not target:
            raise ValueError("请填写目标化学式")
        parse_formula(target)
        if not evidence_linked or not str(source_id or "").strip():
            raise ValueError("请先填写可追踪的资料来源，再使用模型筛选候选实验")

        temperatures = _numeric_options(temperature_options, "候选温度", 0.0, 2500.0)
        durations = _numeric_options(duration_options, "候选时长", 0.05, 500.0)
        atmospheres = _safe_list(atmosphere_options)
        if not atmospheres:
            raise ValueError("请至少选择一种候选气氛")
        combinations = list(itertools.product(temperatures, durations, atmospheres))
        if len(combinations) > 120:
            raise ValueError("候选组合超过120组，请减少温度、时长或气氛选项")

        repeats = max(1, min(int(repeats or 1), 10))
        priority_count = max(1, min(int(priority_count or 1), len(combinations)))
        characterization_hours = max(0.0, min(float(characterization_hours or 0.0), 100.0))
        precursor_list = _safe_list(precursors)
        action_names = _safe_list(actions)
        modalities_list = _safe_list(modalities)
        model = get_model()
        candidates: list[dict[str, Any]] = []

        for temperature, duration, atmosphere in combinations:
            candidate_actions = _make_actions(action_names, temperature, duration, atmosphere)
            record = {
                "target_formula": target,
                "precursor_formulas": precursor_list,
                "route_available": bool(action_names),
                "actions": candidate_actions,
                "chemistry_family": str(family or "unknown"),
                "observed_modalities": modalities_list,
                "source": {
                    "source_id": str(source_id).strip(),
                    "evidence_level": "E2",
                    "measurement_status": "reported" if modalities_list else "not_available",
                    "publication_year": 2025,
                    "observed": True,
                    "is_simulated": False,
                },
            }
            prediction = model.predict(record)
            condition_score = model.score_condition(record)
            target_value = float(condition_score.get("condition_target_score", 0.0))
            other_value = float(condition_score.get("condition_competitor_score", 1.0))
            timed_steps = sum(
                1 for action in candidate_actions
                if action.get("name") in {"HeatingOperation", "DryingOperation", "Wait"}
            )
            process_hours = float(duration) * max(timed_steps, 1)
            candidates.append(
                {
                    "temperature": float(temperature),
                    "duration": float(duration),
                    "atmosphere": str(atmosphere),
                    "target": target_value,
                    "other": float(other_value),
                    "score": target_value,
                    "route_target_prior": float(prediction.get("target_probability_proxy", 0.0)),
                    "condition_score_kind": str(condition_score.get("score_kind") or "unknown"),
                    "feature_support": dict(condition_score.get("feature_support") or {}),
                    "risk": float(prediction.get("evidence_risk", 1.0)),
                    "abstain": bool(prediction.get("abstain")),
                    "hours_per_run": process_hours + characterization_hours,
                }
            )

        candidates.sort(key=lambda item: (not item["abstain"], item["score"]), reverse=True)
        usable = [item for item in candidates if not item["abstain"]]
        if not usable:
            rows = [
                [
                    index,
                    f"{item['temperature']:g}",
                    f"{item['duration']:g}",
                    item["atmosphere"],
                    "—",
                    "—",
                    "暂不排序",
                    "补充可追踪资料后再比较",
                ]
                for index, item in enumerate(candidates, start=1)
            ]
            message = (
                "## 当前资料不足，模型暂不排列实验优先级\n\n"
                "候选组合已经生成，但当前来源或材料类别缺少足够参考记录。"
                "请先补充资料来源，再重新运行筛选。"
            )
            return message, rows, _empty_3d_figure("实验工作量对比", "等待可用的模型排序结果"), None

        selected, model_separates_conditions = _choose_first_round(
            candidates, min(priority_count, len(usable))
        )
        selected_ids = {id(item) for item in selected}
        baseline_runs = len(candidates) * repeats
        guided_runs = len(selected) * repeats
        baseline_hours = sum(item["hours_per_run"] * repeats for item in candidates)
        guided_hours = sum(item["hours_per_run"] * repeats for item in selected)
        saved_runs = baseline_runs - guided_runs
        saved_hours = max(0.0, baseline_hours - guided_hours)
        saved_percent = 100.0 * saved_hours / baseline_hours if baseline_hours else 0.0

        atmosphere_zh = {
            "air": "空气",
            "inert": "惰性气氛",
            "vacuum": "真空",
            "O2": "氧气",
            "Ar": "氩气",
            "N2": "氮气",
        }
        temperatures_all = [float(item["temperature"]) for item in usable]
        durations_all = [float(item["duration"]) for item in usable]
        selected_reasons: dict[int, str] = {}
        for index, item in enumerate(selected):
            selected_reasons[id(item)] = _selection_reason(
                item,
                selected[:index],
                temperatures_all,
                durations_all,
                model_separates_conditions,
            )
        score_order = sorted(usable, key=lambda item: float(item["score"]), reverse=True)
        score_rank = {
            id(item): 1 + sum(float(other["score"]) > float(item["score"]) + 1e-12 for other in score_order)
            for item in usable
        }
        ordered_candidates = selected + [item for item in candidates if id(item) not in selected_ids]
        rows: list[list[Any]] = []
        for index, item in enumerate(ordered_candidates, start=1):
            if item["abstain"]:
                decision = "资料不足"
                reason = "补充可追踪资料后再比较"
            elif id(item) in selected_ids:
                decision = "第一轮先做"
                reason = selected_reasons[id(item)]
            else:
                decision = "等待首轮结果"
                reason = "与已选条件较接近；首轮结果回来后再决定是否需要"
            rows.append(
                [
                    index,
                    f"{item['temperature']:g}",
                    f"{item['duration']:g}",
                    atmosphere_zh.get(item["atmosphere"], item["atmosphere"]),
                    f"{100 * item['score']:.2f}",
                    f"第 {score_rank.get(id(item), '—')} / {len(usable)}",
                    decision,
                    reason,
                ]
            )

        selected_lines = "\n".join(
            f"{index}. **{item['temperature']:g} °C · {item['duration']:g} h · "
            f"{atmosphere_zh.get(item['atmosphere'], item['atmosphere'])}** — "
            f"模型排序分 {100 * item['score']:.2f}，在 {len(usable)} 组中排第 {score_rank[id(item)]}"
            for index, item in enumerate(selected, start=1)
        )
        feature_support = dict(selected[0].get("feature_support") or {})
        supported_names = [
            name for key, name in (("temperature", "温度"), ("duration", "时长"))
            if feature_support.get(key)
        ]
        atmosphere_note = (
            "空气、惰性气氛等差别已经参与模型评分。"
            if feature_support.get("atmosphere_category")
            else (
                "现有训练记录还不能比较空气和惰性气氛的差别，因此相同温度和时长下，两种气氛会得到相同分数。"
                "系统会在首轮中保留不同气氛，方便用实验补齐这部分信息。"
            )
        )
        score_values = [100 * float(item["score"]) for item in usable]
        model_comparison = (
            f"模型排序分从 **{min(score_values):.2f}** 到 **{max(score_values):.2f}**，已经随"
            f"{'、'.join(supported_names) if supported_names else '当前可用过程特征'}变化。{atmosphere_note}"
            if model_separates_conditions
            else (
                "当前可用过程特征仍未产生可分辨的条件差异，因此系统只按条件覆盖范围安排第一轮。"
                + atmosphere_note
            )
        )
        message = f"""## 模型指导的第一轮：先做 {len(selected)} 组条件

你输入了 **{len(candidates)} 组候选条件**。系统实际调用 EC-PathWM **{len(candidates)} 次**，逐组读取模型结果并安排第一轮实验。

{model_comparison}

> **这里的分数用于比较同一批候选条件，分数越高代表模型越建议优先查看。** 实验结果以实际测量为准。总览页的 52.7% 反映已有资料对整条路线的判断，因此只显示一次。

{selected_lines}

### 按当前设置估算的工作量

- 每组重复 **{repeats} 次**；每次检测时间按 **{characterization_hours:g} 小时**计算。
- 第一轮完成全部组合：**{baseline_runs} 次实验，约 {baseline_hours:.1f} 设备小时**。
- 模型指导的第一轮：**{guided_runs} 次实验，约 {guided_hours:.1f} 设备小时**。
- 在获得首轮结果之前，少安排 **{saved_runs} 次实验，少占用约 {saved_hours:.1f} 设备小时（{saved_percent:.1f}%）**。

### 模型如何参与实验过程

1. 根据温度、时长和气氛生成所有候选组合。
2. EC-PathWM 读取每组温度和时长，给出排序分。分数很接近时，系统会优先选择差别较大的条件，扩大首轮信息覆盖范围。
3. 第一轮只完成推荐条件及重复实验，暂缓相近条件。
4. 将新的 XRD、Raman 或 SEM 结果补回系统，再决定第二轮是否需要扩展条件。

### 估算使用的条件

设备小时包含工艺步骤中标注时长的加热、干燥或等待过程，以及填写的检测时间。这里展示首轮排产可以减少多少工作。首轮结果不足时，再从候选表中选择第二轮条件。实际排产时间还会受到设备容量、升降温和样品准备时间影响。

下载的 CSV 已经预留样品编号、XRD 物相、Raman/SEM 记录和实验结论栏，可直接作为首轮实验记录表。
"""
        plan_path = _write_screening_plan_csv(
            target=target,
            precursors=precursor_list,
            source_id=str(source_id).strip(),
            actions=action_names,
            rows=rows,
        )
        return (
            message,
            rows,
            _screening_figure(baseline_runs, guided_runs, baseline_hours, guided_hours),
            plan_path,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return (
            f"## 无法生成筛选计划\n\n{error}",
            [],
            _empty_3d_figure("实验工作量对比", "请先检查候选条件"),
            None,
        )


def _write_report(payload: Mapping[str, Any]) -> str:
    report_id = f"pathmat_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:8]}"
    md_path = REPORT_DIR / f"{report_id}.md"
    json_path = REPORT_DIR / f"{report_id}.json"
    decision = str(payload["decision_markdown"])
    body = (
        "# PATH-Mat Research Console 研究决策报告\n\n"
        f"生成时间：{datetime.now(timezone.utc).isoformat()}\n\n"
        f"{decision}\n\n"
        "## 各模块运行结果\n\n"
        "| 模块 | 状态 | 本次完成的工作 |\n|---|---|---|\n"
        + "\n".join("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in payload["trace_rows"])
        + "\n\n## 模型详细数据\n\n```json\n"
        + json.dumps(payload.get("prediction"), ensure_ascii=False, indent=2)
        + "\n```\n\n## 完整运行记录\n\n```json\n"
        + json.dumps(payload.get("events"), ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    md_path.write_text(body, encoding="utf-8")
    json_path.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(md_path)


def analyze(
    question: str,
    target: str,
    precursors: Any,
    source_id: str,
    family: str,
    actions: Any,
    temperature: float,
    duration: float,
    atmosphere: str,
    modalities: Any,
    license_status: str,
    evidence_linked: bool,
    protocol_complete: bool,
    device_allows: bool,
    challenge: str,
    uploaded_file: str | None = None,
) -> tuple[str, str, str, str, list[list[str]], dict[str, Any], dict[str, Any], str, go.Figure, go.Figure]:
    try:
        uploaded = _parse_uploaded_file(uploaded_file)
        override = _uploaded_override(uploaded)
        target = str(override.get("target") or target or "").strip()
        precursor_list = _safe_list(override.get("precursors") or precursors)
        source_id = str(override.get("source_id") or source_id or "").strip()
        family = str(override.get("family") or family or "unknown").strip()
        action_names = _safe_list(override.get("actions") or actions)
        temperature = float(override.get("temperature") or temperature or 0.0)
        duration = float(override.get("duration") or duration or 0.0)
        atmosphere = str(override.get("atmosphere") or atmosphere or "").strip()
        modalities_list = _safe_list(modalities)

        if challenge == "移除来源":
            evidence_linked = False
            source_id = ""
        elif challenge == "缺少关键表征":
            modalities_list = [item for item in modalities_list if item not in {"XRD", "Raman", "SEM"}]

        formula_ok = True
        formula_error = ""
        try:
            conserved = parse_formula(target)
        except Exception as exc:
            formula_ok = False
            formula_error = f"{type(exc).__name__}: {exc}"
            conserved = {"record": 1.0}

        source_for_record = source_id or "unknown"
        record = {
            "target_formula": target,
            "precursor_formulas": precursor_list,
            "route_available": bool(action_names),
            "actions": _make_actions(action_names, temperature, duration, atmosphere),
            "chemistry_family": family,
            "observed_modalities": modalities_list,
            "source": {
                "source_id": source_for_record,
                "evidence_level": "E2" if evidence_linked else "E0",
                "measurement_status": "reported" if modalities_list else "not_available",
                "publication_year": 2025,
                "observed": bool(evidence_linked),
                "is_simulated": False,
            },
        }

        memory = MemoryStore()
        if evidence_linked and source_id:
            memory.append(
                "evidence_memory",
                {"source_id": source_id, "license_status": license_status},
                source_ids=(source_id,),
                claim_status="verified",
            )

        brain_kwargs: dict[str, Any] = {}
        if challenge == "请求物理执行":
            brain_kwargs["planner"] = PhysicalRequestPlanner()
        brain = CentralBrainAgentV2(
            memory=memory,
            ec_pathwm_adapter=get_model(),
            simulation_enabled=False,
            **brain_kwargs,
        )
        task = BrainTask(
            task_id=f"web-{uuid.uuid4().hex[:12]}",
            objective=question.strip() or "Evaluate a source-bound material route",
            target=target or "unknown",
            source_ids=(source_id,) if source_id else (),
            initial_state={"conserved_elements": conserved},
            metadata={
                "ec_pathwm_record": record,
                "adversarial_input": challenge == "对抗输入",
            },
        )
        result = brain.run(task)
        brain_status = result.status
        prediction: dict[str, Any] | None = None
        if isinstance(result.answer, Mapping):
            maybe_prediction = result.answer.get("ec_pathwm_prediction")
            if isinstance(maybe_prediction, Mapping):
                prediction = dict(maybe_prediction)
        # EC-PathWM abstention is enforced during GATE. In that path the
        # controller returns a bounded reason instead of promoting the model
        # observation into the answer, while the SCORE event still preserves
        # the raw proxy output for audit and user explanation.
        if prediction is None:
            for event in result.events:
                if event.get("stage") != "SCORE":
                    continue
                maybe_prediction = (event.get("payload") or {}).get("ec_pathwm")
                if isinstance(maybe_prediction, Mapping):
                    prediction = dict(maybe_prediction)
                    break

        evidence_level = EvidenceLevel.E2 if evidence_linked and source_id else EvidenceLevel.E0
        package = EvidencePackage(
            claim="Retrospective route-level proxy; no causal or physical execution claim.",
            target=target or "unknown",
            source_ids=[source_id] if source_id else [],
            evidence_level=evidence_level,
            assumptions=["public-record retrospective proxy"],
            tool_calls=["EC-PathWM"],
            uncertainty={
                "evidence_risk": float((prediction or {}).get("evidence_risk", 1.0)),
                "epistemic": float((prediction or {}).get("uncertainty_epistemic_proxy", 1.0)),
            },
            falsification_test="Use source-linked XRD/Raman/SEM and matched controls.",
            safety_constraints=["physical actuation disabled", "plan-only output"],
            license_status=license_status,
        )
        critic = critic_check(package)

        allowed_atmospheres = ["air", "inert", "vacuum"]
        if device_allows and atmosphere.lower() not in allowed_atmospheres:
            allowed_atmospheres.append(atmosphere.lower())
        device = device_gate(
            {"temperature_C": temperature, "duration_h": duration, "atmosphere": atmosphere},
            DeviceLimits(allowed_atmospheres=tuple(allowed_atmospheres)),
        )
        checks = {
            key: bool(protocol_complete)
            for key in (
                "precursor_exists", "purity_checked", "device_compatible", "phase_checked",
                "volatility_checked", "toxicity_checked", "pressure_checked", "atmosphere_checked",
                "uncertainty_checked", "alternative_path_checked",
            )
        }
        protocol = protocol_gate(checks)
        controller_reason = result.answer.get("reason", brain_status) if isinstance(result.answer, Mapping) else brain_status
        controller_reasons = (
            [str(item) for item in controller_reason]
            if isinstance(controller_reason, (list, tuple))
            else [str(controller_reason)]
        )
        controller_gate = {
            "allowed": brain_status not in {"policy_blocked", "external_dependency"},
            "reasons": [] if brain_status not in {"policy_blocked", "external_dependency"} else controller_reasons,
        }

        model_stats_ok = bool(
            prediction
            and float(prediction.get("evidence_risk", 1.0)) <= 0.35
            and float(prediction.get("uncertainty_epistemic_proxy", 1.0)) <= 0.35
        )
        model_gate = {
            "allowed": bool(prediction and not prediction.get("abstain")),
            "reasons": list((prediction or {}).get("abstention_reasons", [])) or ([] if prediction else ["no_model_observation"]),
        }
        gates = {
            "Evidence/Critic": {"allowed": critic.allowed and formula_ok, "reasons": list(critic.reasons) + ([] if formula_ok else [formula_error])},
            "EC-PathWM": model_gate,
            "Device": asdict(device),
            "Protocol": asdict(protocol),
            "Controller/Safety": controller_gate,
        }

        recommendation, rec_reasons, recommendation_trigger, top_next_action = _recommendation(
            modalities_list, prediction, brain_status, gates
        )
        decision = _decision_markdown(
            question, target, source_id, precursor_list, action_names, temperature,
            duration, atmosphere, modalities_list, prediction, recommendation, rec_reasons,
            recommendation_trigger, top_next_action, gates, brain_status,
        )
        hypotheses = _hypothesis_markdown(
            target, modalities_list, prediction, recommendation_trigger
        )
        trace_rows = _trace_rows(
            source_present=bool(evidence_linked and source_id),
            license_ok=license_status.lower() not in {"unknown", "commercial-restricted"},
            record_ok=formula_ok,
            device_allowed=device.allowed,
            critic_allowed=critic.allowed and formula_ok,
            stats_ok=model_stats_ok,
            protocol_allowed=protocol.allowed,
            brain_status=brain_status,
            prediction=prediction,
        )
        route_figure = _route_landscape_figure(prediction)
        agent_figure = _agent_network_figure(trace_rows)

        if brain_status == "policy_blocked":
            status_html = _status_html("blocked", "当前请求包含设备操作", "系统完成资料分析，并停止了设备执行部分。")
        elif brain_status == "abstained" or (prediction and prediction.get("abstain")):
            status_html = _status_html("abstained", "需要补充资料", "当前缺少来源、相近材料记录或关键表征。")
        elif not all(gate["allowed"] for gate in gates.values()):
            status_html = _status_html("review_required", "分析完成，但还有检查项没有通过", "先处理下方红色检查项，再解释模型结果。")
        else:
            target_value = float((prediction or {}).get("target_probability_proxy", 0.0))
            competitor_value = max(((prediction or {}).get("competitor_probabilities") or {"none": 0.0}).values())
            margin_pp = 100 * (target_value - float(competitor_value))
            if abs(margin_pp) < 10:
                status_html = _status_html("review_required", "分析完成：两种可能结果很接近", f"模型对目标产物和其他产物的判断相差 {abs(margin_pp):.1f} 个百分点，需要实验进一步确认。")
            else:
                status_html = _status_html("completed", "分析完成：可以进入下一步实验", "系统已根据现有资料生成检查结果和实验建议。")

        events = list(result.events)
        raw = {
            "record": record,
            "prediction": prediction,
            "brain_status": brain_status,
            "metrics": result.metrics,
            "gates": gates,
            "recommendation": {
                "title": recommendation,
                "trigger": recommendation_trigger,
                "top_next_action_proxy": top_next_action,
                "reasons": rec_reasons,
            },
        }
        report_payload = {
            "decision_markdown": decision,
            "trace_rows": trace_rows,
            "prediction": prediction,
            "events": events,
            "gates": gates,
            "record": record,
        }
        report_path = _write_report(report_payload)
        return (
            status_html,
            _gate_html(gates),
            decision,
            hypotheses,
            trace_rows,
            raw,
            {"events": events},
            report_path,
            route_figure,
            agent_figure,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        status_html = _status_html("blocked", "输入未能完成分析", error)
        trace = [["Orchestrator Agent", "流程编排", "错误", error, "final_status"]]
        return (
            status_html,
            _gate_html({"Input": {"allowed": False, "reasons": [error]}}),
            f"## 输入错误\n\n`{error}`",
            "",
            trace,
            {"error": error},
            {"events": []},
            "",
            _empty_3d_figure("路线风险地形", "输入未完成分析"),
            _empty_3d_figure("系统协作流程", "输入未完成分析"),
        )


def health_check() -> dict[str, Any]:
    model = get_model()
    return {
        "status": "ok",
        "model": type(model.model).__name__,
        "adapter": model.version,
        "model_path": str(MODEL_PATH),
        "agent_contract_count": len(default_agent_contracts()),
    }


__all__ = [
    "PRESETS",
    "analyze",
    "get_model",
    "health_check",
    "load_preset",
    "screen_experiments",
]
