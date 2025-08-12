from .client import (
    register_metadata,
    upload_model_file,
    update_model_metrics,
    get_models_by_did,
    create_new_version,
    make_manifest,
    save_manifest
)

__all__ = [
    "register_metadata",
    "upload_model_file",
    "update_model_metrics",
    "get_models_by_did",
    "create_new_version",
    "make_manifest",
    "save_manifest"
]

__version__ = "0.1.6"