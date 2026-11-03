from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def make_png() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), (80, 150, 70)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_home_page_is_available():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "Clasificador de hojas de limón" in response.text


def test_health_reports_model_status():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert "model_loaded" in response.json()
        assert response.json()["classes"] == ["diseased", "healthy"]


def test_predict_returns_503_when_model_is_not_configured():
    with TestClient(app) as client:
        app.state.model = None
        response = client.post(
            "/predict",
            files={"file": ("leaf.png", make_png(), "image/png")},
        )
        assert response.status_code == 503


def test_predict_returns_class_and_probabilities_with_test_double():
    class FakeModel:
        def predict(self, batch, verbose=0):
            assert batch.shape == (1, 224, 224, 3)
            assert batch.dtype.name == "float32"
            return [[0.2, 0.8]]

    with TestClient(app) as client:
        app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("leaf.png", make_png(), "image/png")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "healthy"
        assert data["class"] == "Hoja sana"
        assert data["confidence"] == 0.8
        assert set(data["probabilities"]) == {"diseased", "healthy"}


def test_predict_rejects_non_image_content_type():
    class FakeModel:
        def predict(self, batch, verbose=0):
            return [[0.5, 0.5]]

    with TestClient(app) as client:
        app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("notes.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 415


def test_predict_rejects_invalid_image_bytes():
    class FakeModel:
        def predict(self, batch, verbose=0):
            return [[0.5, 0.5]]

    with TestClient(app) as client:
        app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("broken.png", b"not really a png", "image/png")},
        )
        assert response.status_code == 400


def test_predict_rejects_empty_file():
    class FakeModel:
        def predict(self, batch, verbose=0):
            return [[0.5, 0.5]]

    with TestClient(app) as client:
        app.state.model = FakeModel()
        response = client.post(
            "/predict",
            files={"file": ("empty.png", b"", "image/png")},
        )
        assert response.status_code == 400


def test_predict_rejects_oversized_upload(monkeypatch):
    from types import SimpleNamespace
    import app.main as main_module

    class FakeModel:
        def predict(self, batch, verbose=0):
            return [[0.5, 0.5]]

    with TestClient(app) as client:
        app.state.model = FakeModel()
        monkeypatch.setattr(main_module, "settings", SimpleNamespace(max_upload_mb=1, image_size=224))
        response = client.post(
            "/predict",
            files={"file": ("large.png", b"x" * (1024 * 1024 + 1), "image/png")},
        )
        assert response.status_code == 413


def test_prepare_image_converts_grayscale_to_rgb():
    from app.model_service import prepare_image

    buffer = BytesIO()
    Image.new("L", (120, 80), 180).save(buffer, format="PNG")
    batch = prepare_image(buffer.getvalue(), 224)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype.name == "float32"


def test_prepare_image_converts_transparency_to_rgb():
    from app.model_service import prepare_image

    buffer = BytesIO()
    Image.new("RGBA", (80, 120), (30, 140, 60, 100)).save(buffer, format="PNG")
    batch = prepare_image(buffer.getvalue(), 224)
    assert batch.shape == (1, 224, 224, 3)


def test_saved_model_archive_has_expected_two_class_signature():
    import json
    from pathlib import Path
    from zipfile import ZipFile

    model_path = Path(__file__).resolve().parents[1] / "models" / "ResNet50.keras"
    assert model_path.is_file(), "El ZIP de entrega debe incluir el modelo ResNet50.keras"
    with ZipFile(model_path) as archive:
        config = json.loads(archive.read("config.json"))

    def find_layer(node, class_name):
        if isinstance(node, dict):
            if node.get("class_name") == class_name:
                return node
            for value in node.values():
                result = find_layer(value, class_name)
                if result is not None:
                    return result
        elif isinstance(node, list):
            for value in node:
                result = find_layer(value, class_name)
                if result is not None:
                    return result
        return None

    dense = find_layer(config, "Dense")
    preprocessor = find_layer(config, "ResNetImageClassifierPreprocessor")
    assert dense is not None
    assert dense["config"]["units"] == 2
    assert dense["config"]["activation"] == "softmax"
    assert preprocessor is not None
    assert preprocessor["config"]["image_converter"]["config"]["image_size"] == [224, 224]
