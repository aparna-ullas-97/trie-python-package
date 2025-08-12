from typing import Dict, Any, Optional
import os
import hashlib
import platform
import json
import subprocess
import datetime
import joblib


# In the future, replace with: import requests
# and call TRIE backend APIs

def _hash_file(file_path: str) -> str:
    """Generate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def make_manifest(model_path: str, auto_fill: bool = True, save_to: str = None) -> dict:
    """
    Generate a TRIE model manifest from a model file.
    Supports PyTorch (.pth, .pt), TensorFlow/Keras (.h5), and Scikit-learn (.joblib, .pkl).

    Args:
        model_path (str): Path to the model file.
        auto_fill (bool): If True, attempts to extract name, shapes, and dependencies automatically.
        save_to (str): If provided, saves manifest to the given JSON file path.

    Returns:
        dict: Manifest data.
    """
    framework = None
    model_name = None
    input_shape = None
    output_shape = None
    dependencies = {}

    # Framework detection
    if model_path.endswith(".h5"):
        framework = "tensorflow/keras"
        if auto_fill:
            try:
                from tensorflow.keras.models import load_model
                model = load_model(model_path)
                model_name = model.name
                input_shape = str(model.input_shape)
                output_shape = str(model.output_shape)
                dependencies = get_dependencies(framework)
            except Exception as e:
                print(f"[WARN] Could not auto-extract Keras model info: {e}")

    elif model_path.endswith((".pt", ".pth")):
        framework = "pytorch"
        if auto_fill:
            try:
                import torch
                model = torch.load(model_path, map_location="cpu")
                model_name = model.__class__.__name__
                # Shape detection may require a dummy input — skipping unless user provides later
                dependencies = get_dependencies(framework)
            except Exception as e:
                print(f"[WARN] Could not auto-extract PyTorch model info: {e}")

    elif model_path.endswith((".joblib", ".pkl")):
        framework = "scikit-learn"
        if auto_fill:
            try:
                model = joblib.load(model_path)
                model_name = model.__class__.__name__
                dependencies = get_dependencies(framework)
            except Exception as e:
                print(f"[WARN] Could not auto-extract scikit-learn model info: {e}")

    else:
        raise ValueError("Unsupported model file type. Supported: .h5, .pth, .pt, .joblib, .pkl")

    # Build manifest
    manifest = {
        "model_name": model_name or "TODO_MODEL_NAME",
        "framework": framework,
        "task": "TODO_TASK",
        "description": "TODO: Add description",
        "version": "1.0.0",
        "dependencies": dependencies,
        "input_shape": input_shape or "TODO",
        "output_shape": output_shape or "TODO",
        "metrics": {},
        "created_by": "TODO_DEVELOPER_NAME",
        "license": "TODO_LICENSE",
        "tags": [],
        "file_format": os.path.splitext(model_path)[-1],
        "checksum": get_checksum(model_path),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z"
    }

    if save_to:
        with open(save_to, "w") as f:
            json.dump(manifest, f, indent=4)
        print(f"[INFO] Manifest saved to {save_to}")

    return manifest


def get_checksum(file_path):
    """Generate SHA-256 checksum for the file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def get_dependencies(framework):
    """Extract relevant dependencies and versions."""
    pkgs = subprocess.check_output(["pip", "freeze"], text=True).splitlines()
    deps = {}
    if framework == "tensorflow/keras":
        for pkg in pkgs:
            if "tensorflow" in pkg.lower() or "keras" in pkg.lower():
                name, version = pkg.split("==")
                deps[name] = version
    elif framework == "pytorch":
        for pkg in pkgs:
            if "torch" in pkg.lower():
                name, version = pkg.split("==")
                deps[name] = version
    elif framework == "scikit-learn":
        for pkg in pkgs:
            if "scikit-learn" in pkg.lower():
                name, version = pkg.split("==")
                deps[name] = version
    return deps


def save_manifest(manifest: Dict[str, Any], output_path: str = "manifest.json"):
    """Save manifest dictionary to JSON file."""
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Manifest saved to {output_path}")

def register_metadata(metadata: Dict[str, Any]) -> str:
    """
    Register model metadata (no file, no metrics yet).
    Returns a model_id string.
    """
    required_fields = ["name", "description", "category", "did"]
    for field in required_fields:
        if field not in metadata:
            raise ValueError(f"Missing required field: {field}")

    print("[STUB] Registering metadata on TRIE...")
    print(metadata)

    # Simulate model_id from blockchain
    model_id = "model_" + os.urandom(4).hex()
    print(f"[STUB] Created model_id: {model_id}")
    return model_id


def upload_model_file(model_id: str, file_path: str) -> Dict[str, Any]:
    """
    Upload a trained model file and link it to an existing model entry.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[STUB] Uploading file {file_path} for model_id {model_id}...")
    file_size = os.path.getsize(file_path)
    return {"status": "success", "model_id": model_id, "file_size": file_size}


def update_model_metrics(model_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Update performance metrics for an existing model.
    """
    print(f"[STUB] Updating metrics for model_id {model_id}...")
    print(metrics)
    return {"status": "success", "model_id": model_id, "metrics": metrics}


def get_models_by_did(did: str, category: Optional[str] = None) -> Dict[str, Any]:
    """
    Retrieve models owned by a specific DID (optionally filter by category).
    """
    if not did:
        raise ValueError("did is required")

    print(f"[STUB] Fetching models for DID={did} (category={category})...")
    # Simulated dataset
    simulated = [
        {"model_id": "model_ab12", "name": "ExampleModel", "category": "NLP", "did": "did:rubix:111"},
        {"model_id": "model_cd34", "name": "AnotherModel", "category": "CV",  "did": "did:rubix:111"},
        {"model_id": "model_ef56", "name": "ThirdModel",   "category": "NLP", "did": "did:rubix:222"},
    ]
    out = [m for m in simulated if m["did"] == did and (category is None or m["category"] == category)]
    return {"models": out}


def create_new_version(model_id: str, file_path: str, version_note: str) -> Dict[str, Any]:
    """
    Create a new version of an existing model.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print(f"[STUB] Creating new version for model_id {model_id} with note: {version_note}")
    file_size = os.path.getsize(file_path)
    return {"status": "success", "model_id": model_id, "version_note": version_note, "file_size": file_size}