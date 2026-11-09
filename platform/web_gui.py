import base64
import io
import json
import os
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit_antd_components as sac
from PIL import Image
from streamlit_image_comparison import image_comparison

try:
    from matplotlib import cm
except ImportError:
    cm = None

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st_lottie = None

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8001")
PAGE_TITLE = "MEEREIS-SR | 中国海洋大学海冰监测平台"
OUC_BLUE = "#004481"
POLAR_LIGHT_BLUE = "#00AAFF"
MODEL_MODE_OPTIONS = {
    "极地常规监测 (100km -> 25km)": "25km",
    "航道精细监测 (40km -> 10km)": "10km",
}
MODEL_MODE_NAMES = {value: label for label, value in MODEL_MODE_OPTIONS.items()}
HISTORY_COLUMNS = ["operator_name", "timestamp", "filename", "success"]
SUPPORTED_TYPES = ["png", "jpg", "jpeg", "bmp", "tif", "tiff"]
LEFT_LABEL = "Bicubic Grayscale"
RIGHT_LABEL = "SRGAN Grayscale"
SLIDER_NOTE = "注：滑块对比旨在展示模型对海冰拓扑结构与边缘锐度的重建效果。"
BG_IMAGE_URL = "https://images.unsplash.com/photo-1734569927533-6b3a1292d2a0?q=80&w=1740&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
HERO_IMAGE_URL = "https://images.unsplash.com/photo-1517783999520-f068d7431a60?q=80&w=2070&auto=format&fit=crop"
LOTTIE_URL = "https://assets5.lottiefiles.com/packages/lf20_st960u97.json"
LOGO_PATH = Path(__file__).resolve().parent / "logo.svg"
HISTORY_PATH = Path(__file__).resolve().parent / "history.csv"
STORAGE_DIR = Path(__file__).resolve().parent / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
RESULTS_DIR = STORAGE_DIR / "results"


def load_lottieurl(url: str) -> dict | None:
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        return None
    return None


LOTTIE_AI = load_lottieurl(LOTTIE_URL) if st_lottie else None


def get_auth_headers() -> dict[str, str]:
    access_token = st.session_state.get("access_token")
    if not access_token:
        raise PermissionError("登录已失效，请重新登录。")
    return {"Authorization": f"Bearer {access_token}"}


def extract_response_detail(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"

    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, list):
        parts = []
        for item in detail:
            if isinstance(item, dict):
                loc = ".".join(str(part) for part in item.get("loc", []))
                msg = item.get("msg") or item
                parts.append(f"{loc}: {msg}" if loc else str(msg))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    if detail:
        return str(detail)
    return str(payload)


def ensure_submit_response_ok(response: requests.Response) -> None:
    if response.ok:
        return

    detail = extract_response_detail(response)
    st.error(detail)
    if response.status_code == 403:
        raise PermissionError(detail)
    raise RuntimeError(detail)


def api_login(username: str, password: str) -> bool:
    try:
        response = requests.post(
            f"{API_BASE_URL}/login",
            data={"username": username, "password": password},
            timeout=15,
        )
    except requests.ConnectionError:
        st.error(f"无法连接后端服务 {API_BASE_URL}，请确认 FastAPI 已启动。")
        return False
    except requests.Timeout:
        st.error("登录请求超时，请稍后重试。")
        return False
    except requests.RequestException as exc:
        st.error(f"登录请求失败: {exc}")
        return False

    if response.status_code == 401:
        st.error("用户名或密码错误。")
        return False

    try:
        response.raise_for_status()
        token_data = response.json()
    except (requests.HTTPError, ValueError) as exc:
        st.error(f"登录失败: {exc}")
        return False

    access_token = token_data.get("access_token")
    if not access_token:
        st.error("登录失败：后端未返回 access_token。")
        return False

    st.session_state["access_token"] = access_token
    st.session_state["current_user"] = token_data.get("username") or username
    st.session_state["current_role"] = token_data.get("role")
    st.session_state["is_logged_in"] = True
    return True


def logout() -> None:
    for key in ["is_logged_in", "access_token", "current_user", "current_role", "single_result", "batch_results", "batch_zip_bytes"]:
        st.session_state.pop(key, None)


def api_query_status(task_id: str) -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/task_status/{task_id}", headers=get_auth_headers(), timeout=10)
    except requests.ConnectionError as exc:
        raise RuntimeError(f"无法连接后端服务 {API_BASE_URL}，请确认 FastAPI 已启动。") from exc
    except requests.Timeout as exc:
        raise RuntimeError("查询任务状态超时，请稍后重试。") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"查询任务状态失败: {exc}") from exc

    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"后端返回错误: {detail}") from exc

    return response.json()


def api_get_history() -> list[dict]:
    try:
        response = requests.get(f"{API_BASE_URL}/history", headers=get_auth_headers(), timeout=10)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise RuntimeError(f"无法连接后端历史接口: {API_BASE_URL}/history") from exc
    except requests.Timeout as exc:
        raise RuntimeError("查询历史记录超时，请检查 FastAPI 服务状态。") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"查询历史记录失败: {exc}") from exc

    return response.json()


def get_logo_data_uri() -> str | None:
    if not LOGO_PATH.exists():
        return None
    svg_text = LOGO_PATH.read_text(encoding="utf-8")
    encoded = base64.b64encode(svg_text.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def inject_enhanced_styles() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        header[data-testid="stHeader"] {{
            background: transparent !important;
            border-bottom: none !important;
        }}
        header[data-testid="stHeader"] button {{
            color: white !important;
            background-color: rgba(255,255,255,0.15) !important;
        }}
        footer {{
            display: none !important;
        }}
        .main .block-container {{
            padding-top: 2rem !important;
            max-width: 95% !important;
        }}
        .stApp {{
            background-image:
                linear-gradient(rgba(0, 10, 30, 0.6), rgba(0, 10, 30, 0.8)),
                url('{BG_IMAGE_URL}');
            background-size: cover !important;
            background-attachment: fixed !important;
            background-position: center !important;
        }}
        [data-testid="stSidebar"] {{
            background: rgba(0, 20, 50, 0.85) !important;
            backdrop-filter: blur(10px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
            min-width: 260px !important;
            max-width: 280px !important;
        }}
        [data-testid="stSidebar"] * {{
            color: #FFFFFF !important;
        }}
        .stTextInput input {{
            background-color: rgba(8, 20, 36, 0.85) !important;
            color: #EAF6FF !important;
            border: 1px solid rgba(255, 255, 255, 0.15) !important;
        }}
        .stTextInput input::placeholder {{
            color: rgba(234, 246, 255, 0.55) !important;
        }}
        .ant-menu {{
            border-radius: 12px !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            background: rgba(255, 255, 255, 0.03) !important;
            overflow: hidden !important;
        }}
        .ant-menu-item {{
            border-radius: 8px !important;
        }}
        .ant-menu-item-selected {{
            border-radius: 8px !important;
            background: rgba(0, 170, 255, 0.2) !important;
            border-right: 3px solid #4fd1ff !important;
        }}
        .glass-tile, .glass-card {{
            background: rgba(255, 255, 255, 0.06) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 16px !important;
            padding: 24px !important;
            margin-bottom: 20px !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3) !important;
            transition: all 0.2s ease !important;
        }}
        .glass-tile:hover, .glass-card:hover {{
            transform: translateY(-4px) !important;
            border-color: rgba(0, 170, 255, 0.4) !important;
        }}
        h1, h2, h3 {{
            color: #ffffff !important;
            font-weight: 600 !important;
        }}
        p, li {{
            color: rgba(255, 255, 255, 0.82) !important;
            font-weight: 400 !important;
        }}
        .small-note {{
            color: rgba(255, 255, 255, 0.55) !important;
            font-size: 0.85rem !important;
        }}
        .metric-value {{
            color: #4fd1ff !important;
            font-size: 2.4rem !important;
            font-weight: 700 !important;
            line-height: 1.2 !important;
        }}
        .hero-banner {{
            background-image:
                linear-gradient(rgba(0, 10, 30, 0.55), rgba(0, 10, 30, 0.75)),
                url('{HERO_IMAGE_URL}');
            background-size: cover;
            background-position: center;
            border-radius: 24px !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            padding: 40px !important;
            text-align: center !important;
            margin-bottom: 30px !important;
        }}
        .stButton > button,
        .stDownloadButton > button {{
            background: linear-gradient(135deg, #0077ff, #00c6ff) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            transition: all 0.2s ease !important;
        }}
                .stButton > button:hover,
        .stDownloadButton > button:hover {{
            background: linear-gradient(135deg, #0088ff, #00d4ff) !important;
        }}
        /* ↓↓↓ 下拉框暴力修复 ↓↓↓ */
        /* 把所有 popover 里的东西先全部刷成黑字白底 */
        div[data-baseweb="popover"] {{
            background-color: #ffffff !important;
        }}
        div[data-baseweb="popover"] * {{
            color: #000000 !important;
            background-color: transparent !important;
        }}
        /* 再单独把高亮/悬浮项设成冰蓝背景+白字（防止被上面的 * 覆盖） */
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li[aria-selected="true"],
        div[data-baseweb="popover"] [role="option"]:hover,
        div[data-baseweb="popover"] [role="option"][aria-selected="true"] {{
            background-color: #00AAFF !important;
            color: #FFFFFF !important;
        }}
        /* 搜索框也保持黑字 */
        div[data-baseweb="popover"] input {{
            color: #000000 !important;
            background-color: #ffffff !important;
        }}
        div[data-baseweb="popover"] input::placeholder {{
            color: #666666 !important;
        }}
        /* 选择框本体也改回黑字 */
        div[data-baseweb="select"] > div {{
            background-color: rgba(255, 255, 255, 0.9) !important;
            border: 1px solid rgba(0, 0, 0, 0.2) !important;
            color: #000000 !important;
        }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{
            color: #000000 !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("single_result", None)
    st.session_state.setdefault("batch_results", [])
    st.session_state.setdefault("batch_zip_bytes", None)
    st.session_state.setdefault("active_tasks", {})
    st.session_state.setdefault("single_originals", {})
    st.session_state.setdefault("batch_originals", {})
    st.session_state.setdefault("is_logged_in", False)
    st.session_state.setdefault("access_token", None)
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("current_role", None)
    st.session_state.setdefault("model_type", "25km")


def load_history() -> pd.DataFrame:
    columns = [
        "id",
        "operator_name",
        "task_id",
        "task_type",
        "model_version",
        "model_type",
        "status",
        "result_path",
        "error_message",
        "timestamp",
        "created_at",
        "updated_at",
    ]
    try:
        history_rows = api_get_history()
    except RuntimeError as exc:
        st.warning(str(exc))
        return pd.DataFrame(columns=columns)

    history_df = pd.DataFrame(history_rows, columns=columns)
    if "model_version" not in history_df.columns:
        history_df["model_version"] = None
    if "model_type" not in history_df.columns:
        history_df["model_type"] = None
    history_df["model_version"] = history_df["model_version"].fillna(history_df["model_type"]).fillna("25km")
    history_df["model_type"] = history_df["model_type"].fillna(history_df["model_version"]).fillna("25km")
    if not history_df.empty:
        history_df["created_at"] = pd.to_datetime(history_df["created_at"], errors="coerce")
        history_df["updated_at"] = pd.to_datetime(history_df["updated_at"], errors="coerce")
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"], errors="coerce")
        history_df["can_view_result"] = (
            (history_df["status"] == "SUCCESS")
            & history_df["result_path"].fillna("").astype(str).ne("")
        )
    else:
        history_df["can_view_result"] = []
    return history_df


def compute_history_stats(history_df: pd.DataFrame) -> tuple[int, int]:
    total_count = len(history_df.index)
    if history_df.empty:
        return total_count, 0
    timestamps = pd.to_datetime(history_df["timestamp"], errors="coerce")
    today_count = int((timestamps.dt.date == datetime.now().date()).sum())
    return total_count, today_count


def compute_status_counts(history_df: pd.DataFrame) -> dict[str, int]:
    if history_df.empty or "status" not in history_df.columns:
        return {"SUCCESS": 0, "FAILURE": 0, "STARTED": 0}
    return {
        "SUCCESS": int((history_df["status"] == "SUCCESS").sum()),
        "FAILURE": int((history_df["status"] == "FAILURE").sum()),
        "STARTED": int((history_df["status"] == "STARTED").sum()),
    }


def register_active_task(
    task_id: str,
    task_type: str,
    filename: str | None = None,
    filenames: list[str] | None = None,
    model_type: str | None = None,
) -> None:
    st.session_state.setdefault("active_tasks", {})
    st.session_state["active_tasks"][task_id] = {
        "task_id": task_id,
        "task_type": task_type,
        "filename": filename,
        "filenames": filenames or [],
        "model_type": model_type or st.session_state.get("model_type", "25km"),
        "state": "PENDING",
        "status": "排队中",
        "result_path": None,
        "result_paths": [],
        "error": None,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def update_active_task_from_status(task_id: str, task_status: dict) -> dict:
    st.session_state.setdefault("active_tasks", {})
    task_record = st.session_state["active_tasks"].setdefault(
        task_id,
        {
            "task_id": task_id,
            "task_type": "SINGLE",
            "filename": None,
            "filenames": [],
            "model_type": "25km",
            "state": "PENDING",
            "status": "排队中",
            "result_path": None,
            "result_paths": [],
            "error": None,
        },
    )
    task_record["state"] = task_status.get("state", task_record.get("state"))
    task_record["status"] = task_status.get("status", task_record.get("status"))
    task_record["result_path"] = task_status.get("result_path") or task_record.get("result_path")
    task_record["result_paths"] = task_status.get("result_paths") or task_record.get("result_paths") or []
    task_record["results"] = task_status.get("results") or task_record.get("results") or []
    task_record["model_type"] = task_status.get("model_type") or task_record.get("model_type") or "25km"
    task_record["error"] = task_status.get("error") or task_record.get("error")
    task_record["updated_at"] = datetime.now().isoformat(timespec="seconds")
    return task_record


def refresh_active_tasks() -> dict[str, int]:
    counts = {"processing": 0, "success": 0, "failure": 0}
    for task_id in list(st.session_state.get("active_tasks", {}).keys()):
        try:
            task_status = api_query_status(task_id)
        except Exception:
            continue
        task_record = update_active_task_from_status(task_id, task_status)
        state = task_record.get("state")
        if state == "SUCCESS":
            counts["success"] += 1
        elif state == "FAILURE":
            counts["failure"] += 1
        else:
            counts["processing"] += 1
    return counts


def build_single_result_from_path(filename: str, result_path: str, original_image: Image.Image | None = None) -> None:
    result_file = Path(result_path)
    if not result_file.exists():
        raise FileNotFoundError(f"结果文件不存在或前端无权读取: {result_path}")

    sr_hr_image = Image.open(result_file).convert("L")
    if original_image is None:
        scale_width = max(1, sr_hr_image.width // 4)
        scale_height = max(1, sr_hr_image.height // 4)
        original_image = sr_hr_image.resize((scale_width, scale_height), Image.BICUBIC)

    assets = build_comparison_assets(original_image.convert("L"), sr_hr_image)
    st.session_state["single_result_path"] = result_path
    st.session_state["single_result"] = {
        "filename": filename,
        "result_path": result_path,
        "input_lr": assets["input_lr"].copy(),
        "bicubic_gray": assets["bicubic_gray"].copy(),
        "sr_gray": assets["sr_gray"].copy(),
        "pseudo_viridis": assets["pseudo_viridis"].copy(),
        "pseudo_jet": assets["pseudo_jet"].copy(),
        "residual": assets["residual"].copy(),
        "download_image": sr_hr_image.copy(),
    }


def resolve_history_result_path(result_path: object) -> str | None:
    if result_path is None:
        return None

    path_text = str(result_path).strip()
    if path_text.lower() in {"nan", "none", "null"}:
        return None
    if not path_text:
        return None

    try:
        parsed_paths = json.loads(path_text)
    except json.JSONDecodeError:
        return path_text

    if isinstance(parsed_paths, list):
        for item in parsed_paths:
            item_text = str(item).strip()
            if item_text:
                return item_text
        return None

    if isinstance(parsed_paths, str):
        return parsed_paths.strip() or None

    return path_text


def load_history_result_image(result_path: object) -> tuple[Image.Image | None, str | None, str | None]:
    display_path = resolve_history_result_path(result_path)
    if not display_path:
        return None, None, "该任务没有可用的 result_path。"

    result_file = Path(display_path)
    if not result_file.exists():
        return None, display_path, f"结果文件不存在或前端无权读取: {display_path}"

    try:
        image = Image.open(result_file).convert("RGB")
    except OSError as exc:
        return None, display_path, f"结果图片读取失败: {exc}"

    return image, display_path, None


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def save_processing_outputs(original_image: Image.Image, sr_gray_image: Image.Image, filename: str, operator_name: str) -> tuple[str, str]:
    unique_id = uuid.uuid4().hex[:8]
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(filename).name
    lr_path = f"storage/uploads/{unique_id}_{safe_filename}"
    sr_path = f"storage/results/sr_{unique_id}_{safe_filename}"
    original_image.save(Path(__file__).resolve().parent / lr_path)
    sr_gray_image.save(Path(__file__).resolve().parent / sr_path)
    return lr_path, sr_path


def build_results_zip(result_images: list[tuple[str, Image.Image]]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename, image in result_images:
            zip_file.writestr(f"{Path(filename).stem}_jet.png", image_to_png_bytes(image))
    return zip_buffer.getvalue()


def apply_colormap(gray_image: np.ndarray, colormap: int) -> Image.Image:
    colored_bgr = cv2.applyColorMap(gray_image, colormap)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(colored_rgb, mode="RGB")


def apply_scientific_colormap(gray_image: np.ndarray, cmap_name: str) -> Image.Image:
    normalized = gray_image.astype(np.float32) / 255.0
    if cm is not None:
        rgba_image = cm.get_cmap(cmap_name)(normalized)
        rgb_image = (rgba_image[:, :, :3] * 255.0).round().astype(np.uint8)
        return Image.fromarray(rgb_image, mode="RGB")

    cv2_colormap = cv2.COLORMAP_VIRIDIS if cmap_name == "viridis" else cv2.COLORMAP_JET
    return apply_colormap(gray_image, cv2_colormap)


def normalize_to_uint8(image_array: np.ndarray) -> np.ndarray:
    array = image_array.astype(np.float32)
    if array.size == 0:
        return array.astype(np.uint8)

    low, high = np.percentile(array, (1, 99))
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def sr_display_grayscale(sr_image: Image.Image) -> np.ndarray:
    sr_array = np.asarray(sr_image.convert("L"), dtype=np.uint8)
    return normalize_to_uint8(sr_array)


def build_comparison_assets(original_image: Image.Image, sr_image: Image.Image) -> dict[str, Image.Image]:
    original_gray = np.asarray(original_image.convert("L"), dtype=np.uint8)
    sr_gray = sr_display_grayscale(sr_image)
    target_height, target_width = sr_gray.shape
    bicubic_gray = cv2.resize(original_gray, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    input_lr_image = Image.fromarray(original_gray, mode="L")
    bicubic_gray_image = Image.fromarray(bicubic_gray, mode="L")
    sr_gray_pil = Image.fromarray(sr_gray, mode="L")
    pseudo_viridis = apply_scientific_colormap(sr_gray, "viridis")
    pseudo_jet = apply_scientific_colormap(sr_gray, "jet")
    residual = np.abs(sr_gray.astype(np.int16) - bicubic_gray.astype(np.int16))
    residual_visual = normalize_to_uint8(residual)
    residual_image = Image.fromarray(residual_visual, mode="L")
    return {
        "input_lr": input_lr_image,
        "bicubic_gray": bicubic_gray_image,
        "sr_gray": sr_gray_pil,
        "pseudo_viridis": pseudo_viridis,
        "pseudo_jet": pseudo_jet,
        "pseudo_color": pseudo_viridis,
        "residual": residual_image,
    }


def render_hero_header() -> None:
    logo_uri = get_logo_data_uri()
    logo_html = ""
    if logo_uri:
        logo_html = (
            f'<img src="{logo_uri}" alt="OUC Logo" '
            'style="width:92px; height:92px; margin-bottom:18px; '
            'filter: drop-shadow(0 6px 18px rgba(0,0,0,0.28));" />'
        )
    st.markdown(
        f"""
        <div class="hero-banner">
            {logo_html}
            <h1 style="font-size: 3.8rem; margin-bottom: 0;">MEEREIS-SR</h1>
            <p style="font-size: 1.4rem; color: {POLAR_LIGHT_BLUE}; margin-top: 12px;">
                中国海洋大学 · 海冰密集度智能监测平台
            </p>
            <div style="margin-top: 20px; font-size: 1rem; color: #FFFFFF; text-shadow: 0 1px 4px rgba(0,0,0,0.5);">
                基于 SRGAN+CBAM 注意力机制的极地海冰高分辨率重建系统
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_login_page() -> None:
    render_hero_header()
    left_col, center_col, right_col = st.columns([1, 1.1, 1])
    with center_col:
        st.markdown(
            """
            <div class="glass-card">
                <h2 style="margin-top:0;">系统登录</h2>
                <p style="margin-bottom:0;">使用平台账号进入 MEEREIS-SR 业务中台。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("jwt_login_form", clear_on_submit=False):
            username = st.text_input("用户名", placeholder="admin / researcher")
            password = st.text_input("密码", type="password", placeholder="请输入密码")
            submitted = st.form_submit_button("登录", width="stretch")

        if submitted:
            if not username or not password:
                st.error("请输入用户名和密码。")
            elif api_login(username, password):
                st.success("登录成功，正在进入系统...")
                st.rerun()


def render_comparison_block(left_image: Image.Image, right_image: Image.Image, pseudo_color: Image.Image, residual: Image.Image) -> None:
    image_comparison(
        img1=left_image,
        img2=right_image,
        label1=LEFT_LABEL,
        label2=RIGHT_LABEL,
        width=700,
        show_labels=True,
        starting_position=50,
        make_responsive=True,
        in_memory=True,
    )
    st.markdown(f'<div class="small-note">{SLIDER_NOTE}</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.image(pseudo_color, caption="伪彩色增强参考图", width="stretch")
    with col2:
        st.image(residual, caption="残差细节图（SR - Bicubic）", width="stretch")


def show_processing_animation():
    left_col, center_col, right_col = st.columns([1, 1.2, 1])
    with center_col:
        if st_lottie and LOTTIE_AI:
            st_lottie(LOTTIE_AI, height=80, key="ai_scan")
        progress_placeholder = st.empty()
        st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
        st.caption("AI 推理中...")
        st.markdown("</div>", unsafe_allow_html=True)
    return progress_placeholder


def render_comparison_block(
    input_lr: Image.Image,
    bicubic: Image.Image,
    srgan: Image.Image,
    residual: Image.Image,
    pseudo_viridis: Image.Image,
    pseudo_jet: Image.Image,
) -> None:
    st.markdown("### 四维重建质量矩阵")
    grid_cols = st.columns(4)
    with grid_cols[0]:
        st.image(input_lr, caption="Input (LR)", width="stretch")
    with grid_cols[1]:
        st.image(bicubic, caption="Bicubic", width="stretch")
    with grid_cols[2]:
        st.image(
            srgan,
            caption=f"SRGAN ({st.session_state.get('current_result_model_type') or st.session_state.get('model_type', '25km')})",
            width="stretch",
        )
    with grid_cols[3]:
        st.image(residual, caption="Residual Map", width="stretch")

    st.markdown("### 伪彩色辅助视图")
    pseudo_cols = st.columns(2)
    with pseudo_cols[0]:
        st.image(pseudo_viridis, caption="Viridis - 低密集度冰区识别", width="stretch")
    with pseudo_cols[1]:
        st.image(pseudo_jet, caption="Jet - 冰水边界增强", width="stretch")

    st.markdown("### 细节锐度对比")
    image_comparison(
        img1=bicubic,
        img2=srgan,
        label1=LEFT_LABEL,
        label2=RIGHT_LABEL,
        width=700,
        show_labels=True,
        starting_position=50,
        make_responsive=True,
        in_memory=True,
    )
    st.markdown(f'<div class="small-note">{SLIDER_NOTE}</div>', unsafe_allow_html=True)


def render_home_page(total_count: int, today_count: int) -> None:
    render_hero_header()
    history_df = st.session_state.get("history_df", pd.DataFrame())
    status_counts = compute_status_counts(history_df)
    success_count = status_counts["SUCCESS"]
    failure_count = status_counts["FAILURE"]
    started_count = status_counts["STARTED"]

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1])
    with col1:
        st.markdown(
            """
            <div class="glass-tile">
                <div style="color: #FFFFFF; font-size: 1rem; opacity: 0.9;">总任务数</div>
                <div class="metric-value">{}</div>
            </div>
            """.format(total_count),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-tile">
                <div style="color: #FFFFFF; font-size: 1rem; opacity: 0.9;">成功</div>
                <div class="metric-value">{success_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="glass-tile">
                <div style="color: #FFFFFF; font-size: 1rem; opacity: 0.9;">失败</div>
                <div class="metric-value">{failure_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""
            <div class="glass-tile">
                <div style="color: #FFFFFF; font-size: 1rem; opacity: 0.9;">排队中</div>
                <div class="metric-value">{started_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="glass-tile" style="text-align: left; padding: 40px; margin-top: 20px;">
            <h2 style='color:#00f2ff; margin-top:0; border-bottom: 2px solid #00f2ff; display: inline-block; padding-bottom: 5px;'>项目使命</h2>
            <p style='font-size:1.2rem; margin-top: 20px; color: #FFFFFF;'>
                针对海冰密集度低分辨率切片，构建一个可交互、可解释、可追踪的超分辨率重建平台。
            </p>
            <p style='font-size:1.1rem; color: rgba(255,255,255,0.9);'>
                利用 SRGAN+CBAM 恢复海冰拓扑结构与边缘细节，为极地科研与演示答辩提供直观展示能力。
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style='color:#00d4ff;'>核心能力</h3>
                <ul style='list-style-type:none; padding-left:0;'>
                    <li>单张海冰切片实时超分分析</li>
                    <li>批量切片处理与 ZIP 打包导出</li>
                    <li>灰度科学对比与残差细节展示</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            """
            <div class="glass-card">
                <h3 style='color:#00d4ff;'>监测重点</h3>
                <ul style='list-style-type:none; padding-left:0;'>
                    <li>滑块视图展示结构锐化效果</li>
                    <li>伪彩色参考图辅助分布判读</li>
                    <li>自动记录操作日志到 history.csv</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about_page() -> None:
    st.markdown("## 关于平台")
    st.markdown(
        """
        <div class="glass-card">
            <h3>课题背景</h3>
            <p>本系统服务于“基于 SRGAN+CBAM 的 25km 海冰密集度数据超分辨率”毕业设计，重点展示模型对海冰空间结构的重建能力与结果可解释性。</p>
        </div>
        <div class="glass-card">
            <h3>系统功能</h3>
            <p>支持单张处理、批量处理、灰度滑块对比、伪彩色参考、残差细节图、历史记录追踪以及结果压缩包导出。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_history_page(history_df: pd.DataFrame, total_count: int, today_count: int) -> None:
    st.markdown("## 历史任务看板")
    status_counts = compute_status_counts(history_df)
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("总任务数", total_count)
    metric_col2.metric("SUCCESS", status_counts["SUCCESS"])
    metric_col3.metric("FAILURE", status_counts["FAILURE"])
    metric_col4.metric("STARTED", status_counts["STARTED"])

    if history_df.empty:
        st.info("暂无历史任务记录。")
        return

    st.markdown("### 任务总览")
    table_columns = ["operator_name", "task_id", "task_type", "model_version", "status", "created_at"]
    table_df = history_df.copy()
    for column in table_columns:
        if column not in table_df.columns:
            table_df[column] = ""
    table_df = table_df[table_columns].rename(
        columns={
            "operator_name": "操作员",
            "task_id": "任务ID",
            "task_type": "类型",
            "status": "状态",
            "created_at": "时间",
        }
    )
    if "model_version" in table_df.columns:
        table_df = table_df.rename(columns={"model_version": "模型尺度"})
    st.dataframe(table_df, width="stretch", hide_index=True)

    st.markdown("### 🔍 历史任务溯源查看")
    task_ids = history_df["task_id"].dropna().astype(str).tolist() if "task_id" in history_df.columns else []
    if not task_ids:
        st.info("当前历史数据中没有可选择的 task_id。")
        return

    selected_task_id = st.selectbox("选择或搜索 task_id", options=task_ids)
    matched_rows = history_df[history_df["task_id"].astype(str) == selected_task_id]
    if matched_rows.empty:
        st.warning("未找到选中的任务记录。")
        return

    selected_record = matched_rows.iloc[0]
    status = str(selected_record.get("status", "")).upper()
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    detail_col1.metric("任务类型", str(selected_record.get("task_type", "-")))
    detail_col2.metric("任务状态", status or "-")
    detail_col3.metric("操作员", str(selected_record.get("operator_name", "-")))
    st.caption(f"创建时间: {selected_record.get('created_at')} | 任务ID: {selected_task_id}")

    if status == "SUCCESS":
        image, display_path, error = load_history_result_image(selected_record.get("result_path"))
        if error:
            st.error(error)
            return
        st.image(image, caption="超分历史结果", width="stretch")
        if display_path:
            st.caption(f"结果路径: {display_path}")
        return

    if status == "FAILURE":
        error_message = selected_record.get("error_message")
        if error_message is None or str(error_message).strip().lower() in {"", "nan", "none", "null"}:
            error_message = "任务失败，但后端未记录 error_message。"
        st.error(str(error_message))
        return

    if status == "STARTED":
        st.info("该任务正在处理中，暂时没有可展示的历史结果。")
        return

    st.warning(f"该任务当前状态为 {status or 'UNKNOWN'}，暂无可展示结果。")


def render_task_progress(task_id: str) -> dict:
    """Render one consistent polling panel and return the latest task record."""
    task_status = api_query_status(task_id)
    task_record = update_active_task_from_status(task_id, task_status)
    state = str(task_record.get("state") or task_status.get("state") or "PENDING").upper()
    status_text = task_record.get("status") or state

    progress_map = {
        "PENDING": 25,
        "QUEUED": 25,
        "STARTED": 70,
        "SUCCESS": 100,
        "FAILURE": 100,
    }
    progress_value = progress_map.get(state, 45)

    st.markdown("### Task Progress")
    st.caption(f"Task ID: {task_id}")
    st.progress(progress_value)

    if state in {"PENDING", "QUEUED"}:
        st.info(f"QUEUED - {status_text}")
        time.sleep(2)
        st.rerun()
    if state == "STARTED":
        st.info(f"STARTED - {status_text}")
        time.sleep(2)
        st.rerun()
    if state == "SUCCESS":
        st.success(f"SUCCESS - {status_text}")
    elif state == "FAILURE":
        st.error(task_record.get("error") or "Task failed")

    return task_record


def _resolve_result_paths(task_record: dict) -> list[str]:
    result_paths = task_record.get("result_paths") or []
    if isinstance(result_paths, str):
        try:
            parsed = json.loads(result_paths)
            result_paths = parsed if isinstance(parsed, list) else [result_paths]
        except json.JSONDecodeError:
            result_paths = [result_paths]

    results = task_record.get("results") or []
    if not result_paths and isinstance(results, list):
        result_paths = [item.get("result_path") for item in results if isinstance(item, dict) and item.get("result_path")]

    single_path = task_record.get("result_path")
    if single_path and not result_paths:
        result_paths = [single_path]

    return [str(path) for path in result_paths if path]


def _build_result_item(filename: str, result_path: str, original_image: Image.Image | None = None) -> dict:
    result_file = Path(result_path)
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    sr_hr_image = Image.open(result_file).convert("L")
    if original_image is None:
        lr_size = (max(1, sr_hr_image.width // 4), max(1, sr_hr_image.height // 4))
        original_image = sr_hr_image.resize(lr_size, Image.BICUBIC)

    assets = build_comparison_assets(original_image.convert("L"), sr_hr_image)
    return {
        "filename": filename,
        "result_path": result_path,
        "status": "success",
        "model_type": st.session_state.get("model_type", "25km"),
        "input_lr": assets["input_lr"].copy(),
        "bicubic_gray": assets["bicubic_gray"].copy(),
        "sr_gray": assets["sr_gray"].copy(),
        "pseudo_viridis": assets["pseudo_viridis"].copy(),
        "pseudo_jet": assets["pseudo_jet"].copy(),
        "residual": assets["residual"].copy(),
        "download_image": sr_hr_image.copy(),
    }


def _render_result_detail(item: dict, index: int, key_prefix: str) -> None:
    if item.get("status") != "success":
        st.error(f"{item.get('filename', 'unknown')} failed: {item.get('status')}")
        return

    previous_model_type = st.session_state.get("current_result_model_type")
    st.session_state["current_result_model_type"] = item.get("model_type") or st.session_state.get("model_type", "25km")
    render_comparison_block(
        item["input_lr"],
        item["bicubic_gray"],
        item["sr_gray"],
        item["residual"],
        item["pseudo_viridis"],
        item["pseudo_jet"],
    )
    if previous_model_type is None:
        st.session_state.pop("current_result_model_type", None)
    else:
        st.session_state["current_result_model_type"] = previous_model_type
    st.caption(f"Result path: {item.get('result_path')}")

    path = Path(item.get("result_path", ""))
    data = path.read_bytes() if path.exists() else image_to_png_bytes(item["download_image"])
    st.download_button(
        label=f"Download HR PNG - {Path(item['filename']).stem}",
        data=data,
        file_name=f"{Path(item['filename']).stem}_HR.png",
        mime="image/png",
        key=f"{key_prefix}_download_{index}",
    )


def _rebuild_single_result(task_record: dict, original_image: Image.Image | None = None) -> None:
    result_paths = _resolve_result_paths(task_record)
    if not result_paths:
        raise RuntimeError("Task finished without result_path")
    filename = task_record.get("filename") or Path(result_paths[0]).name
    item = _build_result_item(filename, result_paths[0], original_image)
    item["task_id"] = task_record.get("task_id")
    item["model_type"] = task_record.get("model_type") or st.session_state.get("model_type", "25km")
    st.session_state["single_result_path"] = result_paths[0]
    st.session_state["single_result"] = item


def _rebuild_batch_results(task_record: dict) -> None:
    result_paths = _resolve_result_paths(task_record)
    if not result_paths:
        raise RuntimeError("Batch task finished without result_paths")

    filenames = task_record.get("filenames") or []
    originals_by_name = st.session_state.get("batch_originals", {}).get(task_record.get("task_id"), {})
    result_images: list[tuple[str, Image.Image]] = []
    batch_results: list[dict] = []

    for index, result_path in enumerate(result_paths):
        filename = filenames[index] if index < len(filenames) else Path(result_path).name
        original = originals_by_name.get(filename)
        try:
            item = _build_result_item(filename, result_path, original)
            item["model_type"] = task_record.get("model_type") or st.session_state.get("model_type", "25km")
            batch_results.append(item)
            result_images.append((filename, item["download_image"].copy()))
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            batch_results.append({"filename": filename, "result_path": result_path, "status": str(exc)})

    st.session_state["batch_results"] = batch_results
    st.session_state["batch_zip_bytes"] = build_results_zip(result_images) if result_images else None


def _get_latest_active_task(task_type: str) -> dict | None:
    candidates = [
        task for task in st.session_state.get("active_tasks", {}).values()
        if task.get("task_type") == task_type
    ]
    if not candidates:
        return None
    return candidates[-1]


def render_single_image_page(operator_name: str) -> None:
    st.markdown("## Single Image Super-Resolution")
    model_type = st.session_state.get("model_type", "25km")
    st.info(f"当前模式：{MODEL_MODE_NAMES.get(model_type, model_type)}")
    uploaded_file = st.file_uploader(
        "Upload a low-resolution sea-ice image",
        type=SUPPORTED_TYPES,
        accept_multiple_files=False,
        key="single_uploader",
    )

    if st.button(
        "Start Super-Resolution",
        width="stretch",
        key="single_run_unified",
        disabled=uploaded_file is None,
    ):
        st.toast("🚀 正在组织请求数据...")
        if uploaded_file is None:
            st.error("请先上传一张待处理图像。")
            return
        st.session_state["single_result"] = None
        file_bytes = uploaded_file.getvalue()
        try:
            task_info = api_submit_task(file_bytes, uploaded_file.name, operator_name, model_type=model_type)
        except PermissionError as exc:
            st.error(f"Permission denied: {exc}")
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return
        if task_info is None:
            return
        task_id = task_info.get("task_id")
        if not task_id:
            st.error(f"Backend did not return task_id: {task_info}")
            return

        register_active_task(task_id, "SINGLE", filename=uploaded_file.name, model_type=model_type)
        st.session_state.setdefault("single_originals", {})[task_id] = Image.open(io.BytesIO(file_bytes)).convert("L")
        st.success(f"Task submitted: {task_id}")
        st.rerun()

    active_task = _get_latest_active_task("SINGLE")
    if active_task and active_task.get("state") not in {"SUCCESS", "FAILURE"}:
        render_task_progress(active_task["task_id"])

    active_task = _get_latest_active_task("SINGLE")
    if active_task and active_task.get("state") == "SUCCESS" and not st.session_state.get("single_result"):
        original = st.session_state.get("single_originals", {}).get(active_task.get("task_id"))
        try:
            _rebuild_single_result(active_task, original)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(str(exc))

    result = st.session_state.get("single_result")
    if result:
        st.markdown("### Scientific Result View")
        _render_result_detail(result, 1, "single")


def render_batch_page(operator_name: str) -> None:
    st.markdown("## Batch Super-Resolution")
    model_type = st.session_state.get("model_type", "25km")
    st.info(f"当前模式：{MODEL_MODE_NAMES.get(model_type, model_type)}")
    uploaded_files = st.file_uploader(
        "Upload multiple low-resolution sea-ice images",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True,
        key="batch_uploader",
    )

    if st.button(
        "Start Batch Super-Resolution",
        width="stretch",
        key="batch_run_unified",
        disabled=not uploaded_files,
    ):
        st.toast("🚀 正在组织批量请求数据...")
        if not uploaded_files:
            st.error("请先上传至少一张待处理图像。")
            return
        st.session_state["batch_results"] = []
        st.session_state["batch_zip_bytes"] = None

        try:
            task_info = api_submit_batch_task(uploaded_files, operator_name, model_type=model_type)
        except PermissionError as exc:
            st.error(f"Permission denied: {exc}")
            return
        except RuntimeError as exc:
            st.error(str(exc))
            return
        if task_info is None:
            return
        task_id = task_info.get("task_id")
        if not task_id:
            st.error(f"Backend did not return batch task_id: {task_info}")
            return

        filenames = [uploaded_file.name for uploaded_file in uploaded_files]
        register_active_task(task_id, "BATCH", filenames=filenames, model_type=model_type)
        originals = {}
        for uploaded_file in uploaded_files:
            originals[uploaded_file.name] = Image.open(io.BytesIO(uploaded_file.getvalue())).convert("L")
        st.session_state.setdefault("batch_originals", {})[task_id] = originals
        st.success(f"Batch task submitted: {task_id}")
        st.rerun()

    active_task = _get_latest_active_task("BATCH")
    if active_task and active_task.get("state") not in {"SUCCESS", "FAILURE"}:
        render_task_progress(active_task["task_id"])

    active_task = _get_latest_active_task("BATCH")
    if active_task and active_task.get("state") == "SUCCESS" and not st.session_state.get("batch_results"):
        try:
            _rebuild_batch_results(active_task)
        except (FileNotFoundError, OSError, RuntimeError) as exc:
            st.error(str(exc))

    batch_results = st.session_state.get("batch_results", [])
    if not batch_results:
        return

    success_count = sum(1 for item in batch_results if item.get("status") == "success")
    st.caption(f"Batch completed: {success_count} / {len(batch_results)} images")

    if st.session_state.get("batch_zip_bytes"):
        st.download_button(
            label="Download All HR Results as ZIP",
            data=st.session_state["batch_zip_bytes"],
            file_name="meereis_sr_batch_results.zip",
            mime="application/zip",
            key="batch_zip_download_unified",
        )

    st.markdown("### Detailed Batch Results")
    for index, item in enumerate(batch_results, start=1):
        title = f"{index}. {item.get('filename', 'result')}"
        with st.expander(title, expanded=index == 1):
            _render_result_detail(item, index, "batch")


def render_active_task_summary(history_df: pd.DataFrame) -> None:
    try:
        active_counts = refresh_active_tasks()
    except Exception:
        active_counts = {"processing": 0, "success": 0, "failure": 0}

    history_success = int((history_df["status"] == "SUCCESS").sum()) if not history_df.empty and "status" in history_df else 0
    history_failure = int((history_df["status"] == "FAILURE").sum()) if not history_df.empty and "status" in history_df else 0
    history_started = int((history_df["status"].isin(["STARTED", "QUEUED", "PENDING"])).sum()) if not history_df.empty and "status" in history_df else 0

    st.markdown("#### Task Status")
    st.caption(f"Processing: {active_counts['processing'] + history_started}")
    st.caption(f"Completed: {active_counts['success'] + history_success}")
    st.caption(f"Failed: {active_counts['failure'] + history_failure}")


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, page_icon="🛰️", layout="wide")
    inject_enhanced_styles()
    init_state()

    if not st.session_state.get("is_logged_in"):
        render_login_page()
        return

    history_df = load_history()
    st.session_state["history_df"] = history_df
    total_count, today_count = compute_history_stats(history_df)

    with st.sidebar:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=80)
        st.markdown("### MEEREIS-SR")
        st.caption("Arctic sea-ice concentration super-resolution platform")

        st.divider()
        selected = sac.menu(
            [
                sac.MenuItem("Home", icon="house"),
                sac.MenuItem("Single Image SR", icon="image"),
                sac.MenuItem("Batch Processing", icon="layers"),
                sac.MenuItem("History", icon="clock"),
                sac.MenuItem("About", icon="info-circle"),
            ],
            color="#4fd1ff",
            index=0,
            variant="light",
            key="main_menu",
        )

        st.divider()
        st.markdown("#### Account")
        operator_name = st.text_input("Operator", value=st.session_state.get("current_user") or "")
        st.caption(f"User: {st.session_state.get('current_user') or '-'}")
        st.caption(f"Role: {st.session_state.get('current_role') or '-'}")
        if st.button("Logout", width="stretch", key="logout_button_jwt_final"):
            logout()
            st.rerun()

        st.divider()
        current_model_type = st.session_state.get("model_type", "25km")
        mode_labels = list(MODEL_MODE_OPTIONS.keys())
        current_label = MODEL_MODE_NAMES.get(current_model_type, mode_labels[0])
        selected_mode_label = st.radio(
            "🛰️ 监测模式选择",
            options=mode_labels,
            index=mode_labels.index(current_label) if current_label in mode_labels else 0,
            key="model_mode_radio_final",
        )
        st.session_state["model_type"] = MODEL_MODE_OPTIONS[selected_mode_label]
        st.caption(f"Model scale: {st.session_state['model_type']}")

        st.divider()
        st.caption("Table: task_history")
        render_active_task_summary(history_df)

    if selected == "Home":
        render_home_page(total_count, today_count)
    elif selected == "Single Image SR":
        render_hero_header()
        render_single_image_page(operator_name)
    elif selected == "Batch Processing":
        render_hero_header()
        render_batch_page(operator_name)
    elif selected == "History":
        render_hero_header()
        render_history_page(history_df, total_count, today_count)
    elif selected == "About":
        render_hero_header()
        render_about_page()


def api_submit_task(
    file_bytes: bytes,
    filename: str,
    operator_name: str,
    model_type: str | None = None,
    role: str | None = None,
) -> dict | None:
    access_token = st.session_state.get("access_token")
    if not access_token:
        st.error("登录状态失效，请重新登录")
        return None

    files = {"file": (filename, file_bytes)}
    data = {
        "operator_name": operator_name or "anonymous",
        "model_type": model_type or st.session_state.get("model_type", "25km"),
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
    except requests.ConnectionError as exc:
        detail = f"Cannot connect to FastAPI backend: {API_BASE_URL}"
        st.error(detail)
        raise RuntimeError(detail) from exc
    except requests.Timeout as exc:
        detail = "Submit task timed out."
        st.error(detail)
        raise RuntimeError(detail) from exc
    except requests.RequestException as exc:
        detail = f"Submit task request failed: {exc}"
        st.error(detail)
        raise RuntimeError(detail) from exc

    ensure_submit_response_ok(response)
    return response.json()


def api_submit_batch_task(
    uploaded_files: list,
    operator_name: str,
    model_type: str | None = None,
    role: str | None = None,
) -> dict | None:
    access_token = st.session_state.get("access_token")
    if not access_token:
        st.error("登录状态失效，请重新登录")
        return None

    files = [
        ("files", (uploaded_file.name, uploaded_file.getvalue()))
        for uploaded_file in uploaded_files
    ]
    data = {
        "operator_name": operator_name or "anonymous",
        "model_type": model_type or st.session_state.get("model_type", "25km"),
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_batch",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=60,
        )
    except requests.ConnectionError as exc:
        detail = f"Cannot connect to FastAPI backend: {API_BASE_URL}"
        st.error(detail)
        raise RuntimeError(detail) from exc
    except requests.Timeout as exc:
        detail = "Submit batch task timed out."
        st.error(detail)
        raise RuntimeError(detail) from exc
    except requests.RequestException as exc:
        detail = f"Submit batch task request failed: {exc}"
        st.error(detail)
        raise RuntimeError(detail) from exc

    ensure_submit_response_ok(response)
    return response.json()


if __name__ == "__main__":
    main()


