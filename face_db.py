import json
import os
import uuid
import threading
from typing import List, Dict, Any


class FaceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.RLock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        if not os.path.exists(self.db_path):
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

    def _load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.db_path):
            return []
        with open(self.db_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass
        return []

    def _save(self, records: List[Dict[str, Any]]) -> None:
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    def list_people(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = self._load()
            return [
                {
                    "person_id": r.get("person_id"),
                    "name": r.get("name"),
                    "encoding_len": len(r.get("encoding", [])),
                }
                for r in records
            ]

    def add_person(self, name: str, encoding: List[float]) -> Dict[str, Any]:
        if not name:
            raise ValueError("name 不能为空")
        if not encoding:
            raise ValueError("encoding 不能为空")
        with self._lock:
            records = self._load()
            person = {
                "person_id": uuid.uuid4().hex,
                "name": name,
                "encoding": list(map(float, encoding)),
            }
            records.append(person)
            self._save(records)
            return person

    def delete_person(self, person_id: str) -> Dict[str, Any]:
        with self._lock:
            records = self._load()
            idx = next((i for i, r in enumerate(records) if r.get("person_id") == person_id), -1)
            if idx < 0:
                raise ValueError("人员不存在")
            removed = records.pop(idx)
            self._save(records)
            return removed

    def get_known_encodings(self) -> List[Dict[str, Any]]:
        with self._lock:
            records = self._load()
            return [
                {
                    "person_id": r.get("person_id"),
                    "name": r.get("name"),
                    "encoding": r.get("encoding", []),
                }
                for r in records
            ]