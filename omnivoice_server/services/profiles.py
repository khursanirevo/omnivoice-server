"""
Manages voice cloning profiles on disk.

Profile structure on disk:
  <profile_dir>/
    <profile_id>/
      ref_audio.wav     <- reference audio
      meta.json         <- {"name": str, "ref_text": str|null, "created_at": str}
"""

from __future__ import annotations

import fcntl
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_META_FILE = "meta.json"
PROFILE_AUDIO_FILE = "ref_audio.wav"
PROFILE_EMBEDDING_FILE = "embedding.pt"


class ProfileNotFoundError(Exception):
    pass


class ProfileAlreadyExistsError(Exception):
    pass


class ProfileService:
    def __init__(self, profile_dir: Path) -> None:
        self._dir = profile_dir

    def _acquire_lock(self, profile_id: str):
        """Get file object for advisory lock. Caller must close it."""
        lock_path = self._dir / f".lock-{profile_id}"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = open(lock_path, "w")  # noqa: SIM115
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return lock_file

    def _release_lock(self, lock_file):
        """Release and close the lock file."""
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()

    def list_profiles(self) -> list[dict]:
        """Return list of profile metadata dicts."""
        profiles = []
        for p in sorted(self._dir.iterdir()) if self._dir.exists() else []:
            if p.is_dir():
                meta = self._read_meta(p)
                if meta:
                    profiles.append({"profile_id": p.name, **meta})
        return profiles

    def get_embedding_cache_path(self, profile_id: str) -> Path:
        """Return path to the cached speaker embedding (.pt) for a profile."""
        return self._profile_path(profile_id) / PROFILE_EMBEDDING_FILE

    def get_ref_audio_path(self, profile_id: str) -> Path:
        """Return path to ref audio file. Raises ProfileNotFoundError if missing."""
        path = self._profile_path(profile_id) / PROFILE_AUDIO_FILE
        if not path.exists():
            raise ProfileNotFoundError(f"Profile '{profile_id}' not found")
        return path

    def get_ref_text(self, profile_id: str) -> str | None:
        """Return ref_text from profile metadata, or None."""
        meta = self._read_meta(self._profile_path(profile_id))
        return meta.get("ref_text") if meta else None

    def save_profile(
        self,
        profile_id: str,
        audio_bytes: bytes,
        ref_text: str | None = None,
        overwrite: bool = False,
    ) -> dict:
        """
        Save a new profile. Raises ProfileAlreadyExistsError if exists and overwrite=False.
        Returns the saved metadata dict.
        """
        lock = self._acquire_lock(profile_id)
        try:
            profile_path = self._profile_path(profile_id)
            if profile_path.exists() and not overwrite:
                raise ProfileAlreadyExistsError(
                    f"Profile '{profile_id}' already exists. Use overwrite=true to replace."
                )

            profile_path.mkdir(parents=True, exist_ok=True)

            # Invalidate stale embedding whenever profile content changes
            embedding = profile_path / PROFILE_EMBEDDING_FILE
            if embedding.exists():
                embedding.unlink()

            # Write audio
            audio_path = profile_path / PROFILE_AUDIO_FILE
            audio_path.write_bytes(audio_bytes)

            # Write metadata
            now = datetime.now(timezone.utc).isoformat()
            meta = {
                "name": profile_id,
                "ref_text": ref_text,
                "created_at": now,
            }
            (profile_path / PROFILE_META_FILE).write_text(
                json.dumps(meta, ensure_ascii=False, indent=2)
            )

            logger.info(f"Saved profile '{profile_id}'")
            return {"profile_id": profile_id, **meta}
        finally:
            self._release_lock(lock)

    def delete_profile(self, profile_id: str) -> None:
        lock = self._acquire_lock(profile_id)
        try:
            profile_path = self._profile_path(profile_id)
            if not profile_path.exists():
                raise ProfileNotFoundError(f"Profile '{profile_id}' not found")
            shutil.rmtree(profile_path)
            logger.info(f"Deleted profile '{profile_id}'")
        finally:
            self._release_lock(lock)

    def _profile_path(self, profile_id: str) -> Path:
        # Sanitize: only allow alphanumeric + dash + underscore
        safe = "".join(c for c in profile_id if c.isalnum() or c in "-_")
        if not safe:
            raise ValueError(f"Invalid profile_id: '{profile_id}'")
        return self._dir / safe

    def _read_meta(self, profile_path: Path) -> dict | None:
        meta_file = profile_path / PROFILE_META_FILE
        if not meta_file.exists():
            return None
        try:
            return json.loads(meta_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
