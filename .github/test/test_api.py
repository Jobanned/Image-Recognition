import asyncio
from pathlib import Path

from api.index import app


def test_health_endpoint_is_available() -> None:
    health_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/api/health"
    )
    response = asyncio.run(health_route.endpoint())

    assert response.status_code == 200
    assert response.body == b'{"status":"ok"}'


def test_frontend_route_serves_index() -> None:
    frontend_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/"
    )
    response = asyncio.run(frontend_route.endpoint())

    assert Path(response.path).name == "index.html"
    assert Path(response.path).is_file()


def test_stylesheet_route_serves_css() -> None:
    stylesheet_route = next(
        route for route in app.routes if getattr(route, "path", None) == "/styles.css"
    )
    response = asyncio.run(stylesheet_route.endpoint())

    assert response.media_type == "text/css"
    assert Path(response.path).name == "styles.css"
    assert Path(response.path).is_file()


def test_monkey_assets_are_mounted() -> None:
    monkey_mount = next(
        route for route in app.routes if getattr(route, "path", None) == "/Monkey"
    )

    assert monkey_mount.name == "monkey-assets"
