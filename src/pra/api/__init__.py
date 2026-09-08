# FastAPI 路由 + 审核工作台接口（00 §15 api 目录；总链路 A·1「HTTP 接入」）
#
# 对外公开面：应用工厂 create_app / 执行器 run_review / 响应 DTO ReviewRunResult；
# router 一并导出供测试或子应用挂载。图的懒加载单例在 service 模块（不在此导出，
# 经 run_review 隐式使用）。
from pra.api.app import create_app
from pra.api.routes import router
from pra.api.schemas import ReviewRunResult
from pra.api.service import run_review

__all__ = [
    "create_app",
    "run_review",
    "ReviewRunResult",
    "router",
]
