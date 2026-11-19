"""CVAT REST API client with session and Personal Access Token authentication."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlsplit

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.task_detector import detect_segmentation_type

logger = logging.getLogger(__name__)


class CVATAPIError(RuntimeError):
    """An API error with a safe, user-facing message."""


class CVATClient:
    """Small CVAT REST wrapper. Write operations are never blindly retried."""

    DEFAULT_URL = "http://localhost:8080"
    ONLINE_URL = "https://cvat.edufun.ai"

    def __init__(
        self,
        base_url: str = DEFAULT_URL,
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        access_token: Optional[str] = None,
        auth_mode: Optional[str] = None,
        organization_slug: Optional[str] = None,
        timeout: int = 30,
    ):
        self.base_url = (base_url or self.DEFAULT_URL).rstrip("/")
        self.username = username if username is not None else os.getenv("CVAT_USERNAME", "")
        self.password = password if password is not None else os.getenv("CVAT_PASSWORD", "")
        self.access_token = access_token or os.getenv("CVAT_ACCESS_TOKEN", "")
        self.legacy_token = token or os.getenv("CVAT_TOKEN", "")
        self.auth_mode = auth_mode or ("pat" if self.access_token else "session")
        configured_organization = (
            organization_slug
            if organization_slug is not None
            else os.getenv("CVAT_ORGANIZATION", "")
        )
        self.organization_slug = configured_organization.strip()
        self.timeout = timeout
        self.session = requests.Session()
        # CVAT uses a vendor media type for REST responses. Sending the generic
        # application/json header can make hosted CVAT return HTTP 406.
        self.session.headers.update({"Accept": "application/vnd.cvat+json"})
        if self.organization_slug:
            self.session.headers["X-Organization"] = self.organization_slug

        # Retry only safe reads. A PATCH can succeed server-side even if the
        # response is lost, so retrying it without a fresh deduplication check
        # could create duplicate annotations.
        retry = Retry(
            total=3,
            connect=3,
            read=2,
            status=2,
            backoff_factor=0.4,
            status_forcelist=(429, 502, 503, 504),
            allowed_methods=frozenset({"GET", "HEAD", "OPTIONS"}),
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        return f"{self.base_url}/{path.lstrip('/')}"

    @staticmethod
    def _safe_error(response: requests.Response) -> str:
        status = response.status_code
        if status == 401:
            return "CVAT từ chối xác thực (401). Kiểm tra tài khoản, mật khẩu hoặc Personal Access Token."
        if status == 403:
            return "Tài khoản đã đăng nhập nhưng không có quyền thao tác này (403)."
        if status == 404:
            return "Không tìm thấy endpoint hoặc tài nguyên CVAT (404). Kiểm tra URL, Task và Job."
        detail = ""
        try:
            payload = response.json()
            detail = payload.get("detail") or payload.get("message") or ""
        except (ValueError, AttributeError):
            detail = ""
        return f"CVAT trả về HTTP {status}" + (f": {detail}" if detail else ".")

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.request(
                method,
                self._url(path),
                timeout=kwargs.pop("timeout", self.timeout),
                **kwargs,
            )
        except requests.Timeout as exc:
            raise CVATAPIError("Hết thời gian chờ khi kết nối CVAT. Kiểm tra mạng hoặc URL máy chủ.") from exc
        except requests.ConnectionError as exc:
            raise CVATAPIError(f"Không kết nối được CVAT tại {self.base_url}. Kiểm tra URL và trạng thái máy chủ.") from exc
        except requests.RequestException as exc:
            raise CVATAPIError(f"Lỗi giao tiếp với CVAT: {exc}") from exc

        if response.status_code >= 400:
            endpoint = urlsplit(self._url(path)).path
            raise CVATAPIError(
                f"{self._safe_error(response)} Endpoint: {method.upper()} {endpoint}."
            )
        return response

    # Authentication -------------------------------------------------
    def _set_pat(self) -> None:
        self.session.headers.pop("Authorization", None)
        self.session.headers["Authorization"] = f"Bearer {self.access_token}"

    def _set_legacy_token(self) -> None:
        self.session.headers.pop("Authorization", None)
        self.session.headers["Authorization"] = f"Token {self.legacy_token}"

    def login(self) -> str:
        """Authenticate via PAT or CVAT's login/session endpoint.

        PAT is preferred for integrations. Username/password login keeps the
        server-issued session cookie and CSRF token in this client's session.
        """
        if self.auth_mode == "pat" or self.access_token:
            if not self.access_token:
                raise CVATAPIError("Nhập Personal Access Token của CVAT để đăng nhập.")
            self._set_pat()
            if not self.is_authenticated():
                self.session.headers.pop("Authorization", None)
                raise CVATAPIError("Personal Access Token không hợp lệ, hết hạn hoặc không có quyền đọc CVAT.")
            return "Personal Access Token"

        if self.auth_mode == "legacy_token" or self.legacy_token:
            if not self.legacy_token:
                raise CVATAPIError("Chưa cấu hình CVAT_TOKEN.")
            self._set_legacy_token()
            if not self.is_authenticated():
                self.session.headers.pop("Authorization", None)
                raise CVATAPIError("Legacy CVAT token không hợp lệ. Hãy chuyển sang Personal Access Token.")
            logger.warning("Legacy CVAT token auth is deprecated; prefer a Personal Access Token.")
            return "Legacy token"

        if not self.username or not self.password:
            raise CVATAPIError("Nhập username/password CVAT hoặc chọn Personal Access Token.")

        # CVAT's supported login endpoint creates a session cookie and CSRF
        # cookie. Older servers may additionally return a deprecated API key.
        self.session.headers.pop("Authorization", None)
        response = self._request(
            "POST",
            "/api/auth/login",
            json={"username": self.username, "password": self.password},
            headers={"Content-Type": "application/json"},
        )
        csrf = self.session.cookies.get("csrftoken")
        if csrf:
            self.session.headers["X-CSRFToken"] = csrf
            self.session.headers["Origin"] = self.base_url
            self.session.headers["Referer"] = f"{self.base_url}/"

        payload: Dict[str, Any] = {}
        try:
            payload = response.json()
        except ValueError:
            pass
        if payload.get("key") and not self.session.cookies.get("sessionid"):
            self.legacy_token = str(payload["key"])
            self._set_legacy_token()
            logger.warning("This CVAT server returned a deprecated API key; prefer a Personal Access Token.")

        if not self.is_authenticated():
            raise CVATAPIError("CVAT chưa tạo được session đăng nhập hợp lệ. Kiểm tra tài khoản hoặc phiên bản máy chủ.")
        return "Session"

    def logout(self) -> None:
        """Clear this client's local credentials and cookies."""
        self.session.cookies.clear()
        for name in ("Authorization", "X-CSRFToken", "Origin", "Referer"):
            self.session.headers.pop(name, None)
        self.access_token = ""
        self.legacy_token = ""

    def is_authenticated(self) -> bool:
        try:
            self._request("GET", "/api/users/self", timeout=min(self.timeout, 10))
            return True
        except CVATAPIError as exc:
            if "404" not in str(exc):
                return False
        try:
            self._request("GET", "/api/users", timeout=min(self.timeout, 10))
            return True
        except CVATAPIError:
            return False

    def test_connection(self) -> Dict[str, Any]:
        """Check server reachability and, when configured, authentication."""
        response = self._request("GET", "/api/server/about")
        info: Dict[str, Any]
        try:
            info = response.json()
        except ValueError:
            info = {}
        if self.access_token or self.legacy_token or self.session.cookies.get("sessionid"):
            if not self.is_authenticated():
                raise CVATAPIError("Máy chủ CVAT truy cập được nhưng session/token đã hết hạn hoặc không hợp lệ.")
        return info

    # Paginated resources --------------------------------------------
    def _list_pages(self, path: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        next_url: Optional[str] = self._url(path)
        first_page = True
        while next_url:
            response = self._request(
                "GET",
                next_url,
                params=params if first_page else None,
            )
            data = response.json()
            if isinstance(data, list):
                items.extend(data)
                break
            items.extend(data.get("results", []))
            next_url = data.get("next")
            first_page = False
        return items

    def list_projects(self) -> List[Dict[str, Any]]:
        return self._list_pages("/api/projects")

    def list_tasks(self, project_id: Optional[int] = None) -> List[Dict[str, Any]]:
        params = {"project_id": project_id} if project_id is not None else None
        return self._list_pages("/api/tasks", params=params)

    def get_task(self, task_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/tasks/{int(task_id)}").json()

    def list_jobs(self, task_id: int) -> List[Dict[str, Any]]:
        task = self.get_task(task_id)
        jobs_url = (task.get("jobs") or {}).get("url")
        if jobs_url:
            return self._list_pages(jobs_url)
        return self._list_pages("/api/jobs", params={"task_id": int(task_id)})

    def get_labels(self, task_id: int, task: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        task = task or self.get_task(task_id)
        embedded = task.get("labels")
        if isinstance(embedded, list) and embedded:
            labels = embedded
        elif isinstance(embedded, dict):
            if embedded.get("results"):
                labels = embedded["results"]
            elif embedded.get("url"):
                labels = self._list_pages(embedded["url"])
            else:
                labels = []
        else:
            params: Dict[str, Any] = {"task_id": int(task_id)}
            if task.get("project_id") is not None:
                params["project_id"] = task["project_id"]
            labels = self._list_pages("/api/labels", params=params)

        # Task payloads often contain compact label summaries. Skeleton SVG is
        # needed by the facial-landmark preflight, so hydrate it from the
        # documented label-detail endpoint when the embedded/list item omits it.
        hydrated: List[Dict[str, Any]] = []
        for label in labels:
            if (
                str(label.get("type", "")).casefold() == "skeleton"
                and not str(label.get("svg") or "").strip()
                and label.get("id") is not None
            ):
                try:
                    detail = self._request("GET", f"/api/labels/{int(label['id'])}").json()
                except CVATAPIError as exc:
                    detail = None
                    label = {**label, "_svg_fetch_error": str(exc)}
                if isinstance(detail, dict):
                    label = {**label, **detail}
                    if not str(detail.get("svg") or "").strip():
                        label["_svg_fetch_result"] = "GET detail thành công nhưng CVAT trả SVG rỗng."
            hydrated.append(label)
        return hydrated

    # Task data and annotations --------------------------------------
    def get_task_data_meta(self, task_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/tasks/{int(task_id)}/data/meta").json()

    def get_job_data_meta(self, job_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/jobs/{int(job_id)}/data/meta").json()

    def get_frame_image(self, task_id: int, frame_number: int, quality: str = "original") -> bytes:
        response = self._request(
            "GET",
            f"/api/tasks/{int(task_id)}/data/",
            params={"type": "frame", "number": int(frame_number), "quality": quality},
            headers={"Accept": "image/*"},
        )
        return response.content

    def get_job_frame_image(self, job_id: int, frame_number: int, quality: str = "original") -> bytes:
        response = self._request(
            "GET",
            f"/api/jobs/{int(job_id)}/data",
            params={"type": "frame", "number": int(frame_number), "quality": quality},
        )
        return response.content

    def get_job(self, job_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/jobs/{int(job_id)}").json()

    def get_job_annotations(self, job_id: int) -> Dict[str, Any]:
        return self._request("GET", f"/api/jobs/{int(job_id)}/annotations").json()

    @staticmethod
    def job_annotations_fingerprint(annotations: Dict[str, Any]) -> str:
        """Fingerprint all annotation collections for a stale-preview check."""
        snapshot = {
            key: annotations.get(key, [])
            for key in ("tags", "shapes", "tracks", "intervals")
        }
        encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def delete_all_job_annotations(
        self,
        job_id: int,
        *,
        expected_fingerprint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete all annotation types from one job, while preserving its frames."""
        job_id = int(job_id)
        annotations = self.get_job_annotations(job_id)
        actual_fingerprint = self.job_annotations_fingerprint(annotations)
        if expected_fingerprint is not None and actual_fingerprint != expected_fingerprint:
            raise CVATAPIError(
                "Annotation của Job đã thay đổi sau khi mở xác nhận. "
                "Tải lại danh sách rồi xác nhận xóa lại."
            )

        counts = {
            "deleted_shapes": len(annotations.get("shapes", [])),
            "deleted_tags": len(annotations.get("tags", [])),
            "deleted_tracks": len(annotations.get("tracks", [])),
            "deleted_intervals": len(annotations.get("intervals", [])),
        }
        if not any(counts.values()):
            return {**counts, "annotations": annotations}

        try:
            self._request("DELETE", f"/api/jobs/{job_id}/annotations/")
        except CVATAPIError:
            # A lost response can follow a successful DELETE. Re-read before
            # reporting failure so the user does not blindly repeat the action.
            refreshed = self.get_job_annotations(job_id)
            if any(refreshed.get(key, []) for key in ("tags", "shapes", "tracks", "intervals")):
                raise
            annotations = refreshed
        else:
            annotations = self.get_job_annotations(job_id)

        remaining = {
            key: len(annotations.get(key, []))
            for key in ("tags", "shapes", "tracks", "intervals")
            if annotations.get(key, [])
        }
        if remaining:
            summary = ", ".join(f"{count} {kind}" for kind, count in remaining.items())
            raise CVATAPIError(
                f"CVAT vẫn còn annotation trong Job {job_id} sau khi xóa ({summary}). "
                "Hãy tải lại Job để kiểm tra trạng thái mới nhất."
            )
        return {**counts, "annotations": annotations}

    def put_job_annotations(self, job_id: int, annotations: Dict[str, Any]) -> requests.Response:
        """Replace all annotations. Kept for explicit legacy callers only."""
        return self._request("PUT", f"/api/jobs/{int(job_id)}/annotations", json=annotations)

    def patch_job_annotations(
        self,
        job_id: int,
        annotations: Dict[str, Any],
        action: str = "create",
    ) -> requests.Response:
        return self._request(
            "PATCH",
            f"/api/jobs/{int(job_id)}/annotations",
            params={"action": action},
            json=annotations,
        )

    @staticmethod
    def track_covers_frame(track: Dict[str, Any], frame_number: int) -> bool:
        """Return whether a CVAT track is active or has a keyframe on this frame."""
        keyframes = sorted(
            (shape for shape in (track.get("shapes") or []) if shape.get("frame") is not None),
            key=lambda shape: int(shape["frame"]),
        )
        active = False
        for shape in keyframes:
            keyframe = int(shape["frame"])
            if keyframe > int(frame_number):
                break
            active = not bool(shape.get("outside", False))
            if keyframe == int(frame_number):
                return True
        return active

    def delete_job_frame_annotations(
        self,
        job_id: int,
        frame_number: int,
        *,
        expected_ids: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """Delete standalone shapes and tags on exactly one frame.

        Tracks are rejected because deleting a track object would also remove its
        annotations from other frames. The caller can pass the IDs shown in its
        preview to prevent deleting annotations added or changed after review.
        """
        frame_number = int(frame_number)
        annotations = self.get_job_annotations(job_id)
        shapes = [
            shape for shape in annotations.get("shapes", [])
            if int(shape.get("frame", -1)) == frame_number
        ]
        tags = [
            tag for tag in annotations.get("tags", [])
            if int(tag.get("frame", -1)) == frame_number
        ]
        tracks = [
            track for track in annotations.get("tracks", [])
            if self.track_covers_frame(track, frame_number)
        ]
        if tracks:
            raise CVATAPIError(
                f"Frame {frame_number} có {len(tracks)} track đang hoạt động. "
                "Không xóa để tránh làm mất nhãn ở các frame khác."
            )

        selected = [*shapes, *tags]
        ids = [str(item.get("id")) for item in selected if item.get("id") is not None]
        if len(ids) != len(selected):
            raise CVATAPIError(
                f"Không thể xóa frame {frame_number}: CVAT không trả ID cho mọi annotation."
            )
        if expected_ids is not None and set(ids) != {str(value) for value in expected_ids}:
            raise CVATAPIError(
                f"Annotation của frame {frame_number} đã thay đổi sau khi mở preview. "
                "Tải lại frame rồi xác nhận lại."
            )
        if not selected:
            return {
                "deleted_shapes": 0,
                "deleted_tags": 0,
                "annotations": annotations,
            }

        payload = {
            "version": 0,
            "tags": tags,
            "shapes": shapes,
            "tracks": [],
            "intervals": [],
        }
        try:
            self.patch_job_annotations(job_id, payload, action="delete")
        except CVATAPIError:
            # A lost response can follow a successful delete; confirm server state
            # before reporting failure or allowing a potentially broad retry.
            refreshed = self.get_job_annotations(job_id)
            remaining_ids = {
                str(item.get("id"))
                for item in [*refreshed.get("shapes", []), *refreshed.get("tags", [])]
                if int(item.get("frame", -1)) == frame_number and item.get("id") is not None
            }
            if remaining_ids.intersection(ids):
                raise
            annotations = refreshed
        else:
            annotations = self.get_job_annotations(job_id)

        remaining = [
            item
            for item in [*annotations.get("shapes", []), *annotations.get("tags", [])]
            if int(item.get("frame", -1)) == frame_number
        ]
        remaining_tracks = [
            track for track in annotations.get("tracks", [])
            if self.track_covers_frame(track, frame_number)
        ]
        if remaining or remaining_tracks:
            raise CVATAPIError(
                f"CVAT vẫn trả về {len(remaining)} shape/tag và {len(remaining_tracks)} track "
                f"trên frame {frame_number} sau khi xóa. "
                "Hãy tải lại Job để kiểm tra trạng thái mới nhất."
            )
        return {
            "deleted_shapes": len(shapes),
            "deleted_tags": len(tags),
            "annotations": annotations,
        }

    @staticmethod
    def _shape_fingerprint(shape: Dict[str, Any]) -> str:
        fields = ("type", "frame", "label_id", "points", "occluded", "outside", "rotation", "elements")
        def normalize(value: Any) -> Any:
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, (int, float)):
                return round(float(value), 5)
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items() if key != "id"}
            return value

        stable = {key: normalize(shape.get(key)) for key in fields}
        return json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def append_job_shapes(self, job_id: int, shapes: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Append only new shapes, preserving existing annotations and deduping retries."""
        unique: Dict[str, Dict[str, Any]] = {}
        for shape in shapes:
            unique.setdefault(self._shape_fingerprint(shape), shape)
        existing = self.get_job_annotations(job_id).get("shapes", [])
        existing_keys = {self._shape_fingerprint(shape) for shape in existing}
        new_shapes = [shape for key, shape in unique.items() if key not in existing_keys]
        if not new_shapes:
            return {"added": 0, "duplicates": len(unique)}

        payload = {"version": 0, "tags": [], "shapes": new_shapes, "tracks": [], "intervals": []}
        try:
            self.patch_job_annotations(job_id, payload, action="create")
        except CVATAPIError:
            # A timeout may happen after CVAT has applied the patch. Check the
            # server before surfacing the error so a manual retry stays safe.
            refreshed = self.get_job_annotations(job_id).get("shapes", [])
            refreshed_keys = {self._shape_fingerprint(shape) for shape in refreshed}
            if all(self._shape_fingerprint(shape) in refreshed_keys for shape in new_shapes):
                return {"added": len(new_shapes), "duplicates": len(unique) - len(new_shapes)}
            raise
        return {"added": len(new_shapes), "duplicates": len(unique) - len(new_shapes)}

    @staticmethod
    def _track_fingerprint(track: Dict[str, Any]) -> str:
        """Stable identity for retry-safe video track creation."""
        shape_fields = (
            "type", "frame", "points", "occluded", "outside", "keyframe",
            "z_order", "rotation", "attributes",
        )

        def normalize(value: Any) -> Any:
            if isinstance(value, bool) or value is None:
                return value
            if isinstance(value, (int, float)):
                return round(float(value), 5)
            if isinstance(value, (list, tuple)):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {key: normalize(item) for key, item in value.items() if key != "id"}
            return value

        stable = {
            "label_id": normalize(track.get("label_id")),
            "frame": normalize(track.get("frame")),
            "group": normalize(track.get("group")),
            "shapes": [
                {key: normalize(shape.get(key)) for key in shape_fields}
                for shape in track.get("shapes", [])
            ],
        }
        return json.dumps(stable, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

    def append_job_tracks(self, job_id: int, tracks: Iterable[Dict[str, Any]]) -> Dict[str, int]:
        """Append new CVAT tracks, preserving existing annotations and safe retries."""
        unique: Dict[str, Dict[str, Any]] = {}
        for track in tracks:
            unique.setdefault(self._track_fingerprint(track), track)
        existing = self.get_job_annotations(job_id).get("tracks", [])
        existing_keys = {self._track_fingerprint(track) for track in existing}
        new_tracks = [track for key, track in unique.items() if key not in existing_keys]
        if not new_tracks:
            return {"added": 0, "duplicates": len(unique)}

        payload = {"version": 0, "tags": [], "shapes": [], "tracks": new_tracks, "intervals": []}
        try:
            self.patch_job_annotations(job_id, payload, action="create")
        except CVATAPIError:
            refreshed = self.get_job_annotations(job_id).get("tracks", [])
            refreshed_keys = {self._track_fingerprint(track) for track in refreshed}
            if all(self._track_fingerprint(track) in refreshed_keys for track in new_tracks):
                return {"added": len(new_tracks), "duplicates": len(unique) - len(new_tracks)}
            raise
        return {"added": len(new_tracks), "duplicates": len(unique) - len(new_tracks)}

    # High-level context ----------------------------------------------
    def load_task_context(self, task_id: int, job_id: Optional[int] = None) -> Dict[str, Any]:
        if job_id is not None:
            # Workers may be allowed to read their assigned job while task-level
            # endpoints (for example GET /api/tasks/{id}) are forbidden.
            selected_job = self.get_job(int(job_id))
            if int(selected_job.get("task_id", -1)) != int(task_id):
                raise CVATAPIError(f"Job {job_id} không thuộc Task {task_id}.")
            task = {
                "id": int(task_id),
                "name": selected_job.get("task_name") or f"Task {task_id}",
                "project_id": selected_job.get("project_id"),
                "labels": selected_job.get("labels", []),
                "status": selected_job.get("status"),
            }
            labels = self.get_labels(task_id, task=task)
            meta = self.get_job_data_meta(int(job_id))
            jobs = [selected_job]
            annotations = self.get_job_annotations(int(job_id))
        else:
            task = self.get_task(task_id)
            labels = self.get_labels(task_id, task=task)
            meta = self.get_task_data_meta(task_id)
            jobs = self.list_jobs(task_id)
            selected_job = jobs[0] if jobs else None
            annotations = (
                self.get_job_annotations(int(selected_job["id"])) if selected_job else None
            )

        frames = meta.get("frames", [])
        if selected_job:
            start_frame = int(selected_job.get("start_frame", meta.get("start_frame", 0)))
            stop_frame = int(selected_job.get("stop_frame", meta.get("stop_frame", start_frame + len(frames) - 1)))
            for index, frame in enumerate(frames):
                if frame.get("number") is None:
                    frame["number"] = start_frame + index
            frames = [
                frame for index, frame in enumerate(frames)
                if start_frame <= int(frame.get("number", start_frame + index)) <= stop_frame
            ]

        segmentation_type = detect_segmentation_type(task.get("name", ""), labels)
        label_ids = {lbl["name"]: lbl["id"] for lbl in labels}
        return {
            "task": task,
            "task_id": int(task_id),
            "project_id": task.get("project_id"),
            "labels": labels,
            "label_names": [lbl["name"] for lbl in labels],
            "label_ids": label_ids,
            "meta": meta,
            "frames": frames,
            "jobs": jobs,
            "job": selected_job,
            "job_id": selected_job.get("id") if selected_job else None,
            "annotations": annotations or {"version": 0, "tags": [], "shapes": [], "tracks": [], "intervals": []},
            "segmentation_type": segmentation_type,
            "image_count": len(frames),
            "frame_width": frames[0].get("width") if frames else None,
            "frame_height": frames[0].get("height") if frames else None,
        }
