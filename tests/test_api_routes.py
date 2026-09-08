"""API 路由层（pra/api/routes.py）轻量单测 —— 不碰真库/真图。

POST /api/v1/reviews 用 **monkeypatch run_and_persist**（避免连 MySQL/执行完整图），
断言 200 与响应信封 {run_id, review_decision}；extra 字段 → 422；GET /health → 200。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from pra.api.app import create_app
from pra.domain.models import Decision, RiskLevel, RiskType, ReviewDecision
from helpers import make_case

_CASE = make_case(case_id="CASE_API_001", brand=None)


def _fake_decision() -> ReviewDecision:
    return ReviewDecision(
        decision=Decision.HUMAN_REVIEW,
        risk_level=RiskLevel.HIGH,
        risk_type=[RiskType.POTENTIAL_IP_RISK],
        decision_confidence=0.91,
        policy=["POLICY_3.2"],
        overrides=[],
    )


def test_health_ok():
    """GET /api/v1/health → 200 {"status": "ok"}（不触图/库，恒轻量）。"""
    with TestClient(create_app()) as client:
        resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_create_review_returns_envelope(monkeypatch):
    """POST /api/v1/reviews：monkeypatch run_and_persist → 200，形状 {run_id,
    review_decision}（HTTP 响应形状与切换前一致）。"""

    async def fake_run_and_persist(case):
        assert case.case_id == _CASE.case_id  # 路由把解析后的 ProductReviewCase 传入
        return {"run_id": "RUN_API_001", "decision": _fake_decision()}

    monkeypatch.setattr("pra.api.routes.run_and_persist", fake_run_and_persist)
    payload = _CASE.model_dump(mode="json")
    with TestClient(create_app()) as client:
        resp = client.post("/api/v1/reviews", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"run_id", "review_decision"}
    assert body["run_id"] == "RUN_API_001"
    decision = body["review_decision"]
    assert decision["decision"] == "HUMAN_REVIEW"
    assert decision["risk_level"] == "HIGH"
    assert decision["risk_type"] == ["POTENTIAL_IP_RISK"]
    assert decision["decision_confidence"] == 0.91
    assert decision["policy"] == ["POLICY_3.2"]


def test_create_review_extra_field_422():
    """请求体带未声明字段（extra=forbid）→ 422（不进路由/不触库）。"""
    payload = _CASE.model_dump(mode="json")
    payload["mystery_field"] = "should-be-rejected"
    with TestClient(create_app()) as client:
        resp = client.post("/api/v1/reviews", json=payload)
    assert resp.status_code == 422


def test_create_review_missing_required_field_422():
    """缺必填字段（如 product）→ 422。"""
    payload = _CASE.model_dump(mode="json")
    del payload["product"]
    with TestClient(create_app()) as client:
        resp = client.post("/api/v1/reviews", json=payload)
    assert resp.status_code == 422


def test_run_and_persist_exception_maps_to_500(monkeypatch):
    """run_and_persist 抛异常 → HTTP 500，detail 含人读信息（异常类型+消息）。"""

    async def broken_run_and_persist(case):
        raise RuntimeError("db down")

    monkeypatch.setattr("pra.api.routes.run_and_persist", broken_run_and_persist)
    payload = _CASE.model_dump(mode="json")
    with TestClient(create_app()) as client:
        resp = client.post("/api/v1/reviews", json=payload)
    assert resp.status_code == 500
    assert "RuntimeError" in resp.json()["detail"]
    assert "db down" in resp.json()["detail"]
