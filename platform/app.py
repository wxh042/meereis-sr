import logging
import uuid
from datetime import datetime
from pathlib import Path

import uvicorn
from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.security import OAuth2PasswordRequestForm

try:
    from .dependencies import verify_researcher_permission
    from .database import SessionLocal, TaskHistory, User, init_db
    from .security import create_access_token, verify_password
except ImportError:
    from dependencies import verify_researcher_permission
    from database import SessionLocal, TaskHistory, User, init_db
    from security import create_access_token, verify_password

try:
    from .celery_worker import (
        TASK_UPLOADS_DIR,
        celery_app,
        normalize_model_type,
        process_sr_batch,
        process_sr_image,
        read_uploaded_image,
        run_super_resolution,
    )
except ImportError:
    from celery_worker import (
        TASK_UPLOADS_DIR,
        celery_app,
        normalize_model_type,
        process_sr_batch,
        process_sr_image,
        read_uploaded_image,
        run_super_resolution,
    )


app = FastAPI(title="Sea Ice Super Resolution API")
logger = logging.getLogger(__name__)


def save_upload_for_task(filename: str, image_bytes: bytes) -> Path:
    TASK_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    safe_filename = Path(filename).name
    saved_path = TASK_UPLOADS_DIR / f"{uuid.uuid4().hex}_{safe_filename}"
    saved_path.write_bytes(image_bytes)
    return saved_path


def append_history(
    operator_name: str,
    task_id: str,
    status: str = "QUEUED",
    task_type: str = "SINGLE",
    model_version: str = "25km",
) -> None:
    try:
        with SessionLocal() as db:
            db.add(
                TaskHistory(
                    operator_name=operator_name,
                    task_id=task_id,
                    task_type=task_type,
                    model_version=model_version,
                    status=status,
                    timestamp=datetime.now(),
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                )
            )
            db.commit()
    except Exception:
        logger.exception(
            "TaskHistory insert failed but request will continue: task_id=%s task_type=%s status=%s model_version=%s",
            task_id,
            task_type,
            status,
            model_version,
        )


def validate_model_type(model_type: str) -> str:
    try:
        return normalize_model_type(model_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def enqueue_sr_task(upload: UploadFile, operator_name: str, model_type: str = "25km"):
    model_type = validate_model_type(model_type)
    filename = upload.filename or "uploaded_image"

    try:
        image_bytes = await upload.read()
        read_uploaded_image(image_bytes)
    except ValueError as exc:
        append_history(operator_name, f"upload_failed_{uuid.uuid4().hex}", "FAILURE", "SINGLE", model_type)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        append_history(operator_name, f"upload_failed_{uuid.uuid4().hex}", "FAILURE", "SINGLE", model_type)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    try:
        saved_path = save_upload_for_task(filename, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Save uploaded file failed: {exc}") from exc

    try:
        task = process_sr_image.delay(str(saved_path), operator_name, Path(filename).name, model_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Submit Celery task failed: {exc}") from exc

    append_history(operator_name, task.id, "QUEUED", "SINGLE", model_type)
    return {
        "task_id": task.id,
        "filename": Path(filename).name,
        "model_type": model_type,
        "status": "\u6392\u961f\u4e2d/Processing",
        "input_path": str(saved_path),
    }


init_db()


@app.get("/")
def home():
    return {
        "title": "Sea Ice Super Resolution API",
        "model": "SRGAN + CBAM Generator",
        "predict_endpoint": "/predict",
        "predict_batch_endpoint": "/predict_batch",
        "task_status_endpoint": "/task_status/{task_id}",
        "history_table": "task_history",
    }


@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    with SessionLocal() as db:
        user = db.query(User).filter(User.username == form_data.username).first()

    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={
            "sub": user.username,
            "role": user.role,
        }
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "role": user.role,
    }


@app.get("/history")
def history():
    with SessionLocal() as db:
        rows = db.query(TaskHistory).order_by(TaskHistory.created_at.desc()).all()
        return [
            {
                "id": row.id,
                "operator_name": row.operator_name,
                "task_id": row.task_id,
                "task_type": row.task_type,
                "model_version": row.model_version,
                "model_type": row.model_version,
                "status": row.status,
                "result_path": row.result_path,
                "error_message": row.error_message,
                "timestamp": row.timestamp.isoformat() if row.timestamp else None,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            }
            for row in rows
        ]


@app.post("/predict")
async def predict(
    current_user=Depends(verify_researcher_permission),
    operator_name: str = Form("anonymous"),
    model_type: str = Form("25km"),
    file: UploadFile = File(...),
):
    return await enqueue_sr_task(file, operator_name, model_type)


@app.post("/predict_batch")
async def predict_batch(
    current_user=Depends(verify_researcher_permission),
    operator_name: str = Form("anonymous"),
    model_type: str = Form("25km"),
    files: list[UploadFile] = File(...),
):
    model_type = validate_model_type(model_type)
    saved_paths = []
    filenames = []

    for upload in files:
        filename = upload.filename or "uploaded_image"
        try:
            image_bytes = await upload.read()
            read_uploaded_image(image_bytes)
        except ValueError as exc:
            append_history(operator_name, f"upload_failed_{uuid.uuid4().hex}", "FAILURE", "BATCH", model_type)
            raise HTTPException(status_code=400, detail=f"{filename}: {exc}") from exc
        except Exception as exc:
            append_history(operator_name, f"upload_failed_{uuid.uuid4().hex}", "FAILURE", "BATCH", model_type)
            raise HTTPException(status_code=500, detail=f"{filename}: upload failed: {exc}") from exc

        try:
            saved_path = save_upload_for_task(filename, image_bytes)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{filename}: save uploaded file failed: {exc}") from exc
        saved_paths.append(str(saved_path))
        filenames.append(Path(filename).name)

    try:
        task = process_sr_batch.delay(saved_paths, operator_name, filenames, model_type)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Submit Celery batch task failed: {exc}") from exc

    append_history(operator_name, task.id, "QUEUED", "BATCH", model_type)
    return {
        "task_id": task.id,
        "model_type": model_type,
        "status": "\u6279\u91cf\u4efb\u52a1\u5df2\u5165\u961f/Batch queued",
        "batch_size": len(saved_paths),
        "input_paths": saved_paths,
    }


@app.get("/task_status/{task_id}")
def task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    response = {
        "task_id": task_id,
        "state": task.state,
    }

    if task.state == "PENDING":
        response["status"] = "\u6392\u961f\u4e2d"
    elif task.state == "STARTED":
        response["status"] = "\u5904\u7406\u4e2d"
        if isinstance(task.info, dict):
            response.update(task.info)
    elif task.state == "SUCCESS":
        result = task.result or {}
        response["status"] = "\u5df2\u5b8c\u6210"
        response["result"] = result
        response["result_path"] = result.get("result_path")
        response["result_paths"] = result.get("result_paths")
        response["results"] = result.get("results")
    elif task.state == "FAILURE":
        response["status"] = "\u5931\u8d25"
        response["error"] = str(task.info)
    else:
        response["status"] = task.state
        if task.info:
            response["info"] = str(task.info)

    return response


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
