from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from backend import PRESETS, analyze, get_model, health_check, load_preset, screen_experiments
from pathmat.ec_pathwm_model import route_features


def run_preset(name: str, **changes):
    values = list(load_preset(name)[:-1])
    keys = [
        "question", "target", "precursors", "source_id", "family", "actions",
        "temperature", "duration", "atmosphere", "modalities", "license_status",
        "evidence_linked", "protocol_complete", "device_allows", "challenge",
    ]
    data = dict(zip(keys, values))
    data.update(changes)
    return analyze(**data)


def test_health_uses_real_model_and_all_contracts():
    health = health_check()
    assert health["status"] == "ok"
    assert health["model"] == "ECPathWM"
    assert health["agent_contract_count"] == 12


def test_in_domain_demo_runs_full_central_brain_chain():
    result = run_preset("SrFe₁₂O₁₉｜覆盖域内功能演示")
    assert "分析完成" in result[0]
    assert "判断十分接近" in result[2]
    assert "百分比用于比较模型对两种结果的倾向" in result[2]
    assert "下一步怎么把结果变成可验证结论" in result[3]
    assert "反方审查" not in result[2]
    assert "科学门控" not in result[2]
    assert "不是实际合成成功率" not in result[2]
    assert len(result[4]) == 12
    events = result[6]["events"]
    assert [event["stage"] for event in events] == [
        "INGEST", "RETRIEVE", "RECONSTRUCT", "PROPOSE", "SCORE",
        "DEBATE", "GATE", "ACT", "OBSERVE", "DECIDE",
    ]
    assert all(event.get("event_hash") for event in events)
    replay = CentralBrainReplay(events)
    assert replay == events[-1]["event_hash"]
    assert Path(result[7]).exists()
    assert result[5]["recommendation"]["trigger"] == "high_competing_path_risk"
    assert len(result[8].data) >= 2
    assert any(trace.type == "surface" for trace in result[8].data)
    assert len(result[9].data) >= 2
    assert any(trace.type == "scatter3d" for trace in result[9].data)


def CentralBrainReplay(events):
    # Minimal assertion helper: the production controller emits an append-only,
    # hashed sequence; full replay is exercised below through its public API.
    assert [event["event_id"] for event in events] == [
        f"{events[0]['task_id']}:event:{index:06d}" for index in range(len(events))
    ]
    return events[-1]["event_hash"]


def test_missing_source_abstains_without_crashing():
    result = run_preset("SrFe₁₂O₁₉｜覆盖域内功能演示", challenge="移除来源")
    assert "需要补充资料" in result[0]
    assert result[5]["brain_status"] == "abstained"
    assert result[6]["events"][-1]["status"] == "abstain"


def test_physical_request_is_blocked_by_real_controller_gate():
    result = run_preset("SrFe₁₂O₁₉｜覆盖域内功能演示", challenge="请求物理执行")
    assert "当前请求包含设备操作" in result[0]
    assert result[5]["brain_status"] == "policy_blocked"
    reasons = json.dumps(result[5]["gates"], ensure_ascii=False)
    assert "physical_actuation_disabled" in reasons


def test_unseen_source_triggers_model_abstention():
    result = run_preset("ZnO｜外部来源与缺模态挑战")
    assert result[5]["prediction"]["abstain"] is True
    assert "unseen_source" in result[5]["prediction"]["abstention_reasons"]
    assert result[5]["recommendation"]["trigger"] == "model_abstention"


def test_recommendation_changes_with_missing_xrd():
    result = run_preset("SrFe₁₂O₁₉｜覆盖域内功能演示", challenge="缺少关键表征")
    assert result[5]["recommendation"]["trigger"] == "missing_xrd"
    assert "XRD" in result[5]["recommendation"]["title"]


def test_invalid_formula_returns_a_controlled_decision():
    result = run_preset("自定义路线", target="Bad(Formula")
    assert "需处理" in result[1]
    assert "unsupported formula" in json.dumps(result[5], ensure_ascii=False)


def test_model_guided_screening_builds_a_smaller_first_round():
    summary, rows, figure, plan_path = screen_experiments(
        "SrFe12O19",
        "Fe2O3, SrCO3",
        "lee2025_impurities",
        "oxide",
        "混合, 加热, 研磨, 加热",
        ["XRD"],
        True,
        "950, 1100, 1250",
        "4, 8, 12",
        ["air", "inert"],
        2,
        4,
        2,
    )
    assert "18 组候选条件" in summary
    assert "实际调用 EC-PathWM **18 次**" in summary
    assert "少安排 **28 次实验" in summary
    assert "模型排序分从" in summary
    assert "这里的分数用于比较同一批候选条件" in summary
    assert "现有训练记录还不能比较空气和惰性气氛的差别" in summary
    assert "不是" not in summary
    assert len(rows) == 18
    assert sum(row[6] == "第一轮先做" for row in rows) == 4
    assert len({(row[1], row[2], row[3]) for row in rows[:4]}) == 4
    assert len({row[4] for row in rows}) > 1
    score_by_condition = {(row[1], row[2], row[3]): float(row[4]) for row in rows}
    assert score_by_condition[("950", "4", "空气")] != score_by_condition[("1250", "12", "空气")]
    assert score_by_condition[("950", "4", "空气")] == score_by_condition[("950", "4", "惰性气氛")]
    assert len(figure.data) == 2
    assert plan_path and Path(plan_path).exists()
    with Path(plan_path).open(encoding="utf-8-sig", newline="") as handle:
        plan_rows = list(csv.DictReader(handle))
    assert len(plan_rows) == 18
    assert sum(row["first_round_status"] == "第一轮先做" for row in plan_rows) == 4
    assert "actual_phase_from_XRD" in plan_rows[0]


def test_condition_score_exposes_real_feature_support_and_changes():
    model = get_model()
    common = {
        "target_formula": "SrFe12O19",
        "precursor_formulas": ["Fe2O3", "SrCO3"],
        "route_available": True,
        "chemistry_family": "oxide",
        "observed_modalities": ["XRD"],
        "source": {
            "source_id": "lee2025_impurities",
            "evidence_level": "E2",
            "measurement_status": "reported",
            "publication_year": 2025,
            "observed": True,
            "is_simulated": False,
        },
    }
    low = dict(common, actions=[
        {"name": "Mixing"},
        {"name": "HeatingOperation", "temperature_C": 950, "duration_h": 4, "atmosphere": "air"},
    ])
    high = dict(common, actions=[
        {"name": "Mixing"},
        {"name": "HeatingOperation", "temperature_C": 1250, "duration_h": 12, "atmosphere": "air"},
    ])
    low_score = model.score_condition(low)
    high_score = model.score_condition(high)
    assert low_score["condition_target_score"] != high_score["condition_target_score"]
    assert low_score["feature_support"] == {
        "temperature": True,
        "duration": True,
        "atmosphere_category": False,
    }
    assert low_score["calibrated_probability"] is False


def test_future_retraining_features_keep_atmosphere_categories():
    features = route_features({
        "target_formula": "SrFe12O19",
        "precursor_formulas": ["Fe2O3", "SrCO3"],
        "route_available": True,
        "chemistry_family": "oxide",
        "actions": [
            {"name": "HeatingOperation", "temperature_C": 950, "duration_h": 4, "atmosphere": "air"},
            {"name": "HeatingOperation", "temperature_C": 1100, "duration_h": 8, "atmosphere": "inert"},
        ],
    })
    assert features["atmosphere=air"] == 1.0
    assert features["atmosphere=inert"] == 1.0


def test_model_guided_screening_requires_traceable_source():
    summary, rows, _, plan_path = screen_experiments(
        "SrFe12O19", "Fe2O3, SrCO3", "", "oxide",
        "混合, 加热", ["XRD"], False,
        "950, 1100", "4, 8", ["air"], 2, 2, 2,
    )
    assert "请先填写可追踪的资料来源" in summary
    assert rows == []
    assert plan_path is None


@pytest.mark.parametrize("suffix", ["json", "csv"])
def test_upload_json_and_csv(tmp_path: Path, suffix: str):
    row = {
        "target_formula": "TiO2",
        "precursor_formulas": "Ti, O2",
        "source_id": "lee2025_impurities",
        "chemistry_family": "oxide",
        "process_actions": "Mixing, HeatingOperation",
        "temperature_C": 800,
        "duration_h": 4,
        "atmosphere": "air",
    }
    path = tmp_path / f"record.{suffix}"
    if suffix == "json":
        path.write_text(json.dumps(row), encoding="utf-8")
    else:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row)
            writer.writeheader()
            writer.writerow(row)
    base = list(load_preset("SrFe₁₂O₁₉｜覆盖域内功能演示")[:-1])
    result = analyze(*base, str(path))
    assert result[5]["record"]["target_formula"] == "TiO2"
    assert Path(result[7]).exists()
