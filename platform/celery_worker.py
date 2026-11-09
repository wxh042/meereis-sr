import csv
import io
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from celery import Celery
from PIL import Image, UnidentifiedImageError

try:
    from matplotlib import cm
except ImportError:
    cm = None

try:
    from .database import update_task_history
    from .model import GeneratorV2, GeneratorV4
except ImportError:
    from database import update_task_history
    from model import GeneratorV2, GeneratorV4


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_WEIGHT_PATHS = {
    "25km": Path(os.getenv("SRGAN_25KM_WEIGHTS_PATH", os.getenv("SRGAN_WEIGHTS_PATH", BASE_DIR / "G_v2_epoch_100.pth"))),
    "10km": Path(os.getenv("SRGAN_10KM_WEIGHTS_PATH", BASE_DIR / "G_v4_epoch_100.pth")),
}
HISTORY_PATH = BASE_DIR / "history.csv"
TASK_STORAGE_DIR = BASE_DIR / "task_storage"
TASK_UPLOADS_DIR = TASK_STORAGE_DIR / "uploads"
TASK_RESULTS_DIR = TASK_STORAGE_DIR / "results"
SCALE_FACTOR = 4

REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", REDIS_URL)

celery_app = Celery("meereis_sr", broker=REDIS_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    result_expires=24 * 60 * 60,
)

# Optional local demo transport: real background worker, no Redis installation required.
# Production deployment continues to use Redis by default.
if REDIS_URL == "filesystem://":
    queue_dir = BASE_DIR / "task_storage" / "queue"
    control_dir = queue_dir / "control"
    queue_dir.mkdir(parents=True, exist_ok=True)
    control_dir.mkdir(parents=True, exist_ok=True)
    celery_app.conf.broker_transport_options = {
        "data_folder_in": str(queue_dir), "data_folder_out": str(queue_dir),
        "control_folder": str(control_dir), "polling_interval": 0.2,
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_cache: dict[str, torch.nn.Module] = {}
logger = logging.getLogger(__name__)


def write_task_history(
    task_id: str | None,
    status: str,
    operator_name: str,
    result_path: str | None = None,
    error_message: str | None = None,
    model_version: str | None = None,
) -> None:
    updated = update_task_history(
        task_id=task_id,
        status=status,
        operator_name=operator_name,
        result_path=result_path,
        error_message=error_message,
        model_version=model_version,
    )
    if not updated:
        logger.error("TaskHistory writeback failed: task_id=%s status=%s", task_id, status)


def ensure_history_file() -> None:
    if HISTORY_PATH.exists():
        return

    with HISTORY_PATH.open("w", newline="", encoding="utf-8-sig") as history_file:
        writer = csv.writer(history_file)
        writer.writerow(["operator_name", "timestamp", "filename", "success"])


def append_history(operator_name: str, filename: str, success: bool) -> None:
    ensure_history_file()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with HISTORY_PATH.open("a", newline="", encoding="utf-8-sig") as history_file:
        writer = csv.writer(history_file)
        writer.writerow([operator_name, timestamp, filename, "yes" if success else "no"])


def normalize_model_type(model_type: str | None) -> str:
    normalized = (model_type or "25km").strip().lower()
    if normalized not in MODEL_WEIGHT_PATHS:
        raise ValueError(f"Unsupported model_type: {model_type}. Expected one of {sorted(MODEL_WEIGHT_PATHS)}.")
    return normalized


def extract_generator_state_dict(checkpoint: object) -> object:
    if isinstance(checkpoint, dict) and "G" in checkpoint:
        return checkpoint["G"]
    if isinstance(checkpoint, dict):
        return checkpoint.get("state_dict") or checkpoint.get("model_state_dict") or checkpoint
    return checkpoint


def load_generator(model_type: str = "25km") -> torch.nn.Module:
    model_type = normalize_model_type(model_type)
    weight_path = MODEL_WEIGHT_PATHS[model_type]
    if not weight_path.exists():
        raise FileNotFoundError(f"Model weight file for {model_type} was not found: {weight_path}")

    model_cls = GeneratorV2 if model_type == "25km" else GeneratorV4
    model = model_cls(scale_factor=SCALE_FACTOR).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = extract_generator_state_dict(checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_generator(model_type: str = "25km") -> torch.nn.Module:
    model_type = normalize_model_type(model_type)
    if model_type not in _model_cache:
        _model_cache[model_type] = load_generator(model_type)
    return _model_cache[model_type]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    grayscale_image = image.convert("L")
    image_array = np.asarray(grayscale_image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0).unsqueeze(0).to(device)


def image_to_chw_tensor(image: Image.Image) -> torch.Tensor:
    grayscale_image = image.convert("L")
    image_array = np.asarray(grayscale_image, dtype=np.float32) / 255.0
    return torch.from_numpy(image_array).unsqueeze(0)


def contrast_stretch_uint8(image_array: np.ndarray, low_percentile: float = 1.0, high_percentile: float = 99.0) -> np.ndarray:
    array = image_array.astype(np.float32)
    if array.size == 0:
        return array.astype(np.uint8)

    low, high = np.percentile(array, (low_percentile, high_percentile))
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)

    stretched = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (stretched * 255.0).round().astype(np.uint8)


def tensor_to_uint8_image(output_tensor: torch.Tensor, enhance_contrast: bool = True) -> np.ndarray:
    output_array = output_tensor.squeeze().detach().float().cpu().numpy()

    # The current Generator returns (tanh(out) + 1) / 2, so the expected range is [0, 1].
    # This branch also tolerates older checkpoints that may still emit [-1, 1].
    if output_array.min() < -0.05:
        output_array = (output_array + 1.0) / 2.0

    output_array = np.clip(output_array, 0.0, 1.0)
    uint8_array = (output_array * 255.0).round().astype(np.uint8)
    return contrast_stretch_uint8(uint8_array) if enhance_contrast else uint8_array


def apply_jet_colormap(gray_image: np.ndarray) -> Image.Image:
    colored_bgr = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb, mode="RGB")


def apply_scientific_colormap(gray_image: np.ndarray, cmap_name: str = "jet") -> Image.Image:
    normalized = gray_image.astype(np.float32) / 255.0
    if cm is not None:
        rgba_image = cm.get_cmap(cmap_name)(normalized)
        rgb_image = (rgba_image[:, :, :3] * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb_image, mode="RGB")

    cv2_colormap = cv2.COLORMAP_VIRIDIS if cmap_name == "viridis" else cv2.COLORMAP_JET
    colored_bgr = cv2.applyColorMap(gray_image, cv2_colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb, mode="RGB")


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def read_uploaded_image(image_bytes: bytes) -> Image.Image:
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Uploaded file is not a valid image.") from exc


def read_image_file(file_path: str) -> Image.Image:
    try:
        return Image.open(file_path).convert("L")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Input file is not a valid image.") from exc


def build_batch_tensor(images: list[Image.Image]) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    tensors = []
    original_sizes = []

    for image in images:
        tensor = image_to_chw_tensor(image)
        _, height, width = tensor.shape
        tensors.append(tensor)
        original_sizes.append((height, width))

    max_height = max(height for height, _ in original_sizes)
    max_width = max(width for _, width in original_sizes)
    padded_tensors = []

    for tensor in tensors:
        _, height, width = tensor.shape
        padded_tensors.append(F.pad(tensor, (0, max_width - width, 0, max_height - height)))

    return torch.stack(padded_tensors, dim=0).to(device), original_sizes


def run_super_resolution(image: Image.Image, model_type: str = "25km") -> tuple[Image.Image, Image.Image, Image.Image]:
    input_tensor = image_to_tensor(image)
    generator = get_generator(model_type)

    with torch.no_grad():
        output_tensor = generator(input_tensor)

    sr_gray_array = tensor_to_uint8_image(output_tensor, enhance_contrast=False)
    sr_gray_image = Image.fromarray(sr_gray_array, mode="L")
    sr_jet_image = apply_scientific_colormap(sr_gray_array, "jet")
    sr_viridis_image = apply_scientific_colormap(sr_gray_array, "viridis")
    return sr_gray_image, sr_jet_image, sr_viridis_image


@celery_app.task(bind=True, name="tasks.process_sr_image")
def process_sr_image(
    self,
    file_path: str,
    operator_name: str = "anonymous",
    original_filename: str | None = None,
    model_type: str = "25km",
):
    model_type = normalize_model_type(model_type)
    filename = original_filename or Path(file_path).name
    self.update_state(
        state="STARTED",
        meta={"status": "GPU inference running", "filename": filename, "model_type": model_type},
    )

    try:
        TASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        image = read_image_file(file_path)
        gray_image, jet_image, viridis_image = run_super_resolution(image, model_type=model_type)
        result_path = TASK_RESULTS_DIR / f"{Path(file_path).stem}_HR.png"
        jet_path = TASK_RESULTS_DIR / f"{Path(file_path).stem}_jet.png"
        viridis_path = TASK_RESULTS_DIR / f"{Path(file_path).stem}_viridis.png"
        gray_image.save(result_path)
        jet_image.save(jet_path)
        viridis_image.save(viridis_path)
        append_history(operator_name, filename, True)
        write_task_history(
            task_id=self.request.id,
            status="SUCCESS",
            operator_name=operator_name,
            result_path=str(result_path),
            model_version=model_type,
        )
        return {
            "status": "SUCCESS",
            "filename": filename,
            "model_type": model_type,
            "result_path": str(result_path),
            "preview_path": str(jet_path),
            "viridis_path": str(viridis_path),
        }
    except Exception as exc:
        append_history(operator_name, filename, False)
        write_task_history(
            task_id=self.request.id,
            status="FAILURE",
            operator_name=operator_name,
            error_message=str(exc),
            model_version=model_type,
        )
        raise


@celery_app.task(bind=True, name="tasks.process_sr_batch")
def process_sr_batch(
    self,
    file_paths: list[str],
    operator_name: str = "anonymous",
    original_filenames: list[str] | None = None,
    model_type: str = "25km",
):
    model_type = normalize_model_type(model_type)
    filenames = original_filenames or [Path(file_path).name for file_path in file_paths]
    self.update_state(
        state="STARTED",
        meta={
            "status": "GPU batch inference running",
            "batch_size": len(file_paths),
            "filenames": filenames,
            "model_type": model_type,
        },
    )

    try:
        TASK_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        images = [read_image_file(file_path) for file_path in file_paths]
        batch_tensor, original_sizes = build_batch_tensor(images)
        generator = get_generator(model_type)

        with torch.no_grad():
            output_batch = generator(batch_tensor)

        result_items = []
        for index, output_tensor in enumerate(output_batch):
            height, width = original_sizes[index]
            scaled_height = height * SCALE_FACTOR
            scaled_width = width * SCALE_FACTOR
            cropped_output = output_tensor[:, :scaled_height, :scaled_width]
            sr_gray_array = tensor_to_uint8_image(cropped_output, enhance_contrast=False)
            sr_gray_image = Image.fromarray(sr_gray_array, mode="L")
            sr_jet_image = apply_scientific_colormap(sr_gray_array, "jet")
            result_path = TASK_RESULTS_DIR / f"{Path(file_paths[index]).stem}_HR.png"
            jet_path = TASK_RESULTS_DIR / f"{Path(file_paths[index]).stem}_jet.png"
            sr_gray_image.save(result_path)
            sr_jet_image.save(jet_path)
            append_history(operator_name, filenames[index], True)
            result_items.append(
                {
                    "filename": filenames[index],
                    "model_type": model_type,
                    "result_path": str(result_path),
                    "preview_path": str(jet_path),
                }
            )

        result_paths = [item["result_path"] for item in result_items]
        write_task_history(
            task_id=self.request.id,
            status="SUCCESS",
            operator_name=operator_name,
            result_path=json.dumps(result_paths, ensure_ascii=False),
            model_version=model_type,
        )
        return {
            "status": "SUCCESS",
            "model_type": model_type,
            "batch_size": len(result_items),
            "results": result_items,
            "result_paths": result_paths,
        }
    except Exception as exc:
        for filename in filenames:
            append_history(operator_name, filename, False)
        write_task_history(
            task_id=self.request.id,
            status="FAILURE",
            operator_name=operator_name,
            error_message=str(exc),
            model_version=model_type,
        )
        raise
