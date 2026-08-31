import asyncio

from api.index import app


def test_health_endpoint_is_available() -> None:
    health_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/health"
    )
    response = asyncio.run(health_route.endpoint())

    assert response.status_code == 200
    assert response.body == b'{"status":"ok"}'
