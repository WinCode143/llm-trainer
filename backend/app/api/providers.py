"""Helper models: adding, testing and listing remote models."""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session, select

from ..db import Provider, get_session
from ..services import providers as svc

router = APIRouter(prefix="/api/providers", tags=["providers"])


@router.get("/presets")
def presets():
    """Configuration presets and available families."""
    return {"presets": svc.PRESETS, "kinds": svc.KINDS}


@router.get("")
def listing():
    """Configured providers. Keys come back masked."""
    return svc.listing()


class ProviderBody(BaseModel):
    name: str
    kind: str = "openai"
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    enabled: bool = True
    note: str = ""


@router.post("")
def create(body: ProviderBody, s: Session = Depends(get_session)):
    if s.exec(select(Provider).where(Provider.name == body.name)).first():
        raise HTTPException(409, "There's already a helper with that name")
    p = Provider(**body.model_dump())
    s.add(p)
    s.commit()
    s.refresh(p)
    return svc.listing()


@router.put("/{pid}")
def update(pid: int, body: ProviderBody, s: Session = Depends(get_session)):
    p = s.get(Provider, pid)
    if not p:
        raise HTTPException(404)
    data = body.model_dump()
    # If the interface sends back the masked key (“…abcd”), it wasn't touched: the existing one is kept
    if not data["api_key"] or data["api_key"].startswith("…"):
        data.pop("api_key")
    for k, v in data.items():
        setattr(p, k, v)
    s.add(p)
    s.commit()
    return svc.listing()


@router.delete("/{pid}")
def delete(pid: int, s: Session = Depends(get_session)):
    p = s.get(Provider, pid)
    if p:
        s.delete(p)
        s.commit()
    return {"ok": True}


@router.post("/{pid}/test")
def test(pid: int, s: Session = Depends(get_session)):
    p = s.get(Provider, pid)
    if not p:
        raise HTTPException(404)
    return svc.test(p)


@router.get("/{pid}/models")
def models(pid: int, s: Session = Depends(get_session)):
    p = s.get(Provider, pid)
    if not p:
        raise HTTPException(404)
    try:
        return {"models": svc.remote_models(p)}
    except Exception as e:  # noqa: BLE001  — many servers don't expose /models
        return {"models": [], "error": str(e)[:300]}


@router.get("/choices")
def choices():
    """Options for the “helper model” dropdowns spread across the interface."""
    from ..services.inference import manager
    local = manager.current.name if manager.current else None
    out = [{"id": None, "label": f"Local model ({local})" if local else "Local model (none loaded)",
            "ready": bool(local)}]
    for p in svc.listing():
        if p["enabled"]:
            out.append({"id": p["id"], "label": f"{p['name']} · {p['model']}", "ready": True})
    return out
