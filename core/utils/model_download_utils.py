# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-20 14:38:28
# @Function      : LHM++ auto-download: packages (huggingface_hub, modelscope) and models

import os
import subprocess
import sys
from typing import Dict

sys.path.append('./')

from core.utils.model_card import (
    HuggingFace_MODEL_CARD,
    HuggingFace_Prior_MODEL_CARD,
    ModelScope_MODEL_CARD,
    ModelScope_Prior_MODEL_CARD,
)

# --- Ensure huggingface_hub==0.23.2 to avoid conflicts with modelscope (requires hf<1.0) ---
_HF_HUB_TARGET_VERSION = "0.23.2"
_hf_version_ok = False
try:
    import huggingface_hub as _hf_mod
    _hf_version_ok = getattr(_hf_mod, "__version__", "") == _HF_HUB_TARGET_VERSION
except ImportError:
    pass
if not _hf_version_ok:
    print(f"huggingface_hub is not {_HF_HUB_TARGET_VERSION}. Installing...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", f"huggingface_hub=={_HF_HUB_TARGET_VERSION}"]
        )
        if "huggingface_hub" in sys.modules:
            del sys.modules["huggingface_hub"]
        print(f"huggingface_hub=={_HF_HUB_TARGET_VERSION} installed.")
    except Exception as e:
        print(f"Failed to install huggingface_hub=={_HF_HUB_TARGET_VERSION}: {e}")

# --- Hugging Face Hub Import (auto-install if missing) ---
package_name = "huggingface_hub"
hf_snapshot = None
try:
    from huggingface_hub import snapshot_download as hf_snapshot_import

    hf_snapshot = hf_snapshot_import
    print(f"{package_name} imported successfully.")
except ImportError:
    print(f"{package_name} is not installed. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} has been installed.")
        from huggingface_hub import snapshot_download as hf_snapshot_import

        hf_snapshot = hf_snapshot_import
    except Exception as e:
        print(f"Failed to install or import {package_name}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during {package_name} import: {e}")

# --- ModelScope Import (auto-install if missing) ---
# Note: modelscope requires huggingface-hub<1.0, but project may use hf>=1.x.
# On conflict, we try pip install modelscope --no-deps (snapshot_download often works).
package_name = "modelscope"
ms_snapshot = None
try:
    from modelscope import snapshot_download as ms_snapshot_import

    ms_snapshot = ms_snapshot_import
    print(f"{package_name} imported successfully.")
except ImportError:
    print(f"{package_name} is not installed. Attempting to install...")
    err_msg = None
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        from modelscope import snapshot_download as ms_snapshot_import

        ms_snapshot = ms_snapshot_import
        print(f"{package_name} has been installed.")
    except Exception as e:
        err_msg = str(e)
        if "huggingface" in err_msg.lower() or "huggingface-hub" in err_msg.lower() or "huggingface_hub" in err_msg.lower():
            print(f"Standard install failed (huggingface-hub version conflict). Trying --no-deps...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", package_name, "--no-deps"]
                )
                from modelscope import snapshot_download as ms_snapshot_import

                ms_snapshot = ms_snapshot_import
                print(f"{package_name} has been installed (--no-deps).")
            except Exception as e2:
                print(f"Failed to install or import {package_name}: {e2}")
        else:
            print(f"Failed to install or import {package_name}: {e}")
except Exception as e:
    print(f"An unexpected error occurred during {package_name} import: {e}")


def _is_valid_model_dir(path: str) -> bool:
    """Check if path contains LHM++ or LHM model files (config + weights)."""
    if not path or not os.path.isdir(path):
        return False
    config_files = ("config.json", "configuration.json")
    weight_files = ("pytorch_model.bin", "model.safetensors")
    has_config = any(os.path.exists(os.path.join(path, f)) for f in config_files)
    has_weights = any(os.path.exists(os.path.join(path, f)) for f in weight_files)
    # LHM++ may use sharded safetensors
    if not has_weights:
        has_weights = any(
            f.endswith(".safetensors")
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        )
    return has_config or has_weights


def _get_max_step_folder(current_path: str):
    """
    Find the best model checkpoint path.
    - For LHM: look for step_* folders and return the one with max step number.
    - For LHM++: flat structure (config.json + model.safetensors) -> return current_path.
    """
    if not os.path.isdir(current_path):
        return None

    step_folders = [
        f
        for f in os.listdir(current_path)
        if f.startswith("step_") and os.path.isdir(os.path.join(current_path, f))
    ]

    if not step_folders:
        if _is_valid_model_dir(current_path):
            return current_path
        return None

    def _step_num(name):
        try:
            return int(name.split("_")[1])
        except (IndexError, ValueError):
            return 0

    max_folder = max(step_folders, key=_step_num)
    return os.path.join(current_path, max_folder)


class AutoModelQuery:
    """
    LHM++ auto-download: query model path from local cache or download from
    HuggingFace / ModelScope.
    """

    def __init__(
        self, save_dir: str = "./pretrained_models", hf_kwargs=None, ms_kwargs=None
    ):
        save_dir = os.path.abspath(save_dir)
        # If broken symlink (target missing), remove it and create real dir
        if os.path.lexists(save_dir) and not os.path.exists(save_dir):
            try:
                os.unlink(save_dir)
                print(f"Removed broken symlink, will create real dir: {save_dir}")
            except OSError:
                save_dir = os.path.join(os.path.dirname(save_dir), "pretrained_models_local")
        self.base_save_dir = save_dir
        self.hf_save_dir = os.path.join(self.base_save_dir, "huggingface")
        self.ms_save_dir = self.base_save_dir

        os.makedirs(self.base_save_dir, exist_ok=True)
        os.makedirs(self.hf_save_dir, exist_ok=True)
        os.makedirs(self.ms_save_dir, exist_ok=True)

        self._logger = lambda x: "\033[31m{}\033[0m".format(x)

    def _ensure_trailing_slash(self, path: str) -> str:
        return path + "/" if path and path[-1] != "/" else path

    def query_huggingface_model(self, model_name: str, local_only: bool = False) -> str:
        """Query HuggingFace model, optionally local cache only."""
        if hf_snapshot is None:
            print(self._logger("Hugging Face Hub library not available."))
            raise ImportError("huggingface_hub not imported")

        if model_name not in HuggingFace_MODEL_CARD:
            raise ValueError(
                f"Model '{model_name}' not found in HuggingFace_MODEL_CARD."
            )

        model_repo_id = HuggingFace_MODEL_CARD[model_name]
        action = "Checking cache for" if local_only else "Querying/Downloading"
        print(f"{action} Hugging Face model: {model_repo_id}")

        try:
            model_path = hf_snapshot(
                repo_id=model_repo_id,
                cache_dir=self.hf_save_dir,
                local_files_only=local_only,
            )
            print(
                f"Hugging Face model path {'found locally' if local_only else 'obtained'}: {model_path}"
            )
            return model_path
        except FileNotFoundError:
            if local_only:
                print(
                    f"Hugging Face model {model_repo_id} not found in local cache {self.hf_save_dir}."
                )
            else:
                print(
                    self._logger(f"Cannot download {model_repo_id} from Hugging Face")
                )
            raise
        except Exception as exc:
            log_prefix = "Local check for" if local_only else "Download attempt for"
            print(
                self._logger(
                    f"{log_prefix} Hugging Face model {model_repo_id} failed: {exc}"
                )
            )
            raise exc

    def query_modelscope_model(self, model_name: str, local_only: bool = False) -> str:
        """Query ModelScope model, optionally local cache only."""
        if ms_snapshot is None:
            print(self._logger("ModelScope library not available."))
            raise ImportError("modelscope not imported")

        if model_name not in ModelScope_MODEL_CARD:
            raise ValueError(
                f"Model '{model_name}' not found in ModelScope_MODEL_CARD."
            )

        model_repo_id = ModelScope_MODEL_CARD[model_name]
        expected_ms_base_path = os.path.join(self.ms_save_dir, model_repo_id)

        action = "Checking cache for" if local_only else "Querying/Downloading"
        print(f"{action} ModelScope model: {model_repo_id}")

        if local_only:
            if os.path.isdir(expected_ms_base_path):
                model_path = _get_max_step_folder(expected_ms_base_path)
                if model_path and os.path.isdir(model_path):
                    print(f"Found local ModelScope model at: {model_path}")
                    return model_path
                raise FileNotFoundError(
                    f"Local ModelScope model incomplete or invalid structure for {model_repo_id}"
                )
            raise FileNotFoundError(
                f"Local ModelScope model not found for {model_repo_id}"
            )

        try:
            downloaded_path = ms_snapshot(model_repo_id, cache_dir=self.ms_save_dir)
            model_path = _get_max_step_folder(downloaded_path)
            if model_path and os.path.isdir(model_path):
                print(f"ModelScope model path obtained: {model_path}")
                return model_path
            raise FileNotFoundError(
                f"Failed to resolve model path within ModelScope download for {model_repo_id}"
            )
        except Exception as e:
            print(
                self._logger(
                    f"Download attempt for ModelScope model {model_repo_id} failed: {e}"
                )
            )
            raise e

    def query_prior_huggingface_model(
        self, model_name: str, local_only: bool = False
    ) -> str:
        """Query HuggingFace prior model.
        Uses cache_dir so files go to pretrained_models/huggingface/...; then
        _link_prior_bundle_to_base creates symlinks in pretrained_models/.
        All prior content in pretrained_models/ are symlinks to HF or MS cache.
        """
        if hf_snapshot is None:
            raise ImportError("huggingface_hub not imported")
        if model_name not in HuggingFace_Prior_MODEL_CARD:
            raise ValueError(
                f"Model '{model_name}' not in HuggingFace_Prior_MODEL_CARD"
            )
        repo_id = HuggingFace_Prior_MODEL_CARD[model_name]
        print(f"{'Checking' if local_only else 'Downloading'} HF prior: {repo_id}")
        model_path = hf_snapshot(
            repo_id=repo_id,
            cache_dir=self.hf_save_dir,
            local_files_only=local_only,
        )
        return model_path

    def query_prior_modelscope_model(
        self, model_name: str, local_only: bool = False
    ) -> str:
        """Query ModelScope prior model."""
        if ms_snapshot is None:
            raise ImportError("modelscope not imported")
        if model_name not in ModelScope_Prior_MODEL_CARD:
            raise ValueError(
                f"Model '{model_name}' not in ModelScope_Prior_MODEL_CARD"
            )
        repo_id = ModelScope_Prior_MODEL_CARD[model_name]
        expected_path = os.path.join(self.ms_save_dir, repo_id)
        print(f"{'Checking' if local_only else 'Downloading'} MS prior: {repo_id}")
        if local_only:
            if os.path.isdir(expected_path):
                resolved = _get_max_step_folder(expected_path)
                if resolved and os.path.isdir(resolved):
                    return resolved
                # Prior bundle may have flat structure (no step_*), use path as-is
                return expected_path
            raise FileNotFoundError(f"Local prior model not found: {repo_id}")
        downloaded = ms_snapshot(repo_id, cache_dir=self.ms_save_dir)
        resolved = _get_max_step_folder(downloaded)
        return resolved if resolved else downloaded

    def _link_prior_bundle_to_base(self, prior_path: str) -> None:
        """
        Create symlinks in base_save_dir for each top-level item in the prior bundle.
        Skips items that already exist and have valid targets.
        Removes broken symlinks and re-creates them. Enables flat paths like
        pretrained_models/human_model_files, pretrained_models/voxel_grid, etc.
        """
        prior_path = os.path.normpath(prior_path.rstrip("/"))
        if not os.path.isdir(prior_path):
            return
        base = self.base_save_dir
        for name in os.listdir(prior_path):
            if name.startswith("."):
                continue
            src = os.path.join(prior_path, name)
            dst = os.path.join(base, name)
            if os.path.lexists(dst):
                if os.path.exists(dst):
                    continue
                try:
                    os.unlink(dst)
                    print(f"Removed broken symlink: {name}")
                except OSError as e:
                    print(self._logger(f"Failed to remove broken symlink {name}: {e}"))
                    continue
            try:
                src_rel = os.path.relpath(src, base)
                os.symlink(src_rel, dst)
                print(f"Linked {name} -> prior bundle")
            except OSError as e:
                print(self._logger(f"Failed to link {name}: {e}"))

    def query_prior(self, model_name: str) -> str:
        """
        Prior model query: skip if local exists, else try HuggingFace first, then ModelScope on failure.
        """
        is_hf = model_name in HuggingFace_Prior_MODEL_CARD
        is_ms = model_name in ModelScope_Prior_MODEL_CARD
        if not is_hf and not is_ms:
            raise ValueError(
                f"Prior model '{model_name}' not in any prior model card"
            )

        # 1. Check local HuggingFace cache (cache_dir -> huggingface/, symlinks to base)
        if is_hf:
            try:
                path = self.query_prior_huggingface_model(model_name, local_only=True)
                if path and os.path.isdir(path):
                    print(f"Prior model exists locally (HF): {path}")
                    self._link_prior_bundle_to_base(path)
                    return self._ensure_trailing_slash(self.base_save_dir)
            except (FileNotFoundError, ImportError, Exception) as e:
                print(f"Local HF cache check: {e}")

        # 2. Check local ModelScope cache (cache_dir -> repo subdir, symlinks to base)
        if is_ms:
            try:
                path = self.query_prior_modelscope_model(model_name, local_only=True)
                if path and os.path.isdir(path):
                    print(f"Prior model exists locally (MS): {path}")
                    self._link_prior_bundle_to_base(path)
                    return self._ensure_trailing_slash(self.base_save_dir)
            except (FileNotFoundError, ImportError, Exception) as e:
                print(f"Local MS cache check: {e}")

        # 3. Try download from HuggingFace (cache_dir -> huggingface/, then symlinks to base)
        if is_hf:
            try:
                path = self.query_prior_huggingface_model(
                    model_name, local_only=False
                )
                print(f"Prior model downloaded from HuggingFace: {path}")
                self._link_prior_bundle_to_base(path)
                return self._ensure_trailing_slash(self.base_save_dir)
            except Exception as e:
                print(f"HuggingFace download failed: {e}. Trying ModelScope.")

        # 4. Try download from ModelScope
        if is_ms:
            try:
                path = self.query_prior_modelscope_model(
                    model_name, local_only=False
                )
                print(f"Prior model downloaded from ModelScope: {path}")
                self._link_prior_bundle_to_base(path)
                return self._ensure_trailing_slash(self.base_save_dir)
            except Exception as e:
                raise FileNotFoundError(
                    f"Prior model '{model_name}' download failed: {e}"
                ) from e

        raise FileNotFoundError(
            f"Prior model '{model_name}' could not be obtained or downloaded"
        )

    def download_all_prior_models(self) -> Dict[str, str]:
        """
        Download all prior models to pretrained_models; skip if local exists.
        Returns {model_name: model_path} mapping.
        """
        names = set(HuggingFace_Prior_MODEL_CARD) | set(
            ModelScope_Prior_MODEL_CARD
        )
        result = {}
        for name in names:
            try:
                path = self.query_prior(name)
                result[name] = path
            except Exception as e:
                print(self._logger(f"Prior model {name} failed: {e}"))
        return result

    def query(self, model_name: str) -> str:
        """
        Query model path with prioritized strategy:
        1. Check local ModelScope cache
        2. Check local HuggingFace cache
        3. Download from HuggingFace
        4. Download from ModelScope
        """
        print(f"\n--- Querying model: {model_name} ---")

        is_in_hf = model_name in HuggingFace_MODEL_CARD
        is_in_ms = model_name in ModelScope_MODEL_CARD
        if not is_in_hf and not is_in_ms:
            raise ValueError(
                f"Model name '{model_name}' not found in HuggingFace or ModelScope cards."
            )

        model_path = None

        # 1. Local ModelScope
        if is_in_ms:
            try:
                print("Step 1: Checking local ModelScope cache...")
                model_path = self.query_modelscope_model(model_name, local_only=True)
                if model_path:
                    print(f"Success: Found in local ModelScope cache: {model_path}")
                    return self._ensure_trailing_slash(model_path)
            except FileNotFoundError:
                print("Info: Not found in local ModelScope cache.")
            except ImportError:
                print(
                    self._logger(
                        "Warning: ModelScope library not available for local check."
                    )
                )
            except Exception as e:
                print(
                    self._logger(f"Warning: Error checking local ModelScope cache: {e}")
                )
        else:
            print("Step 1: Skipping local ModelScope check (not in ModelScope card).")

        # 2. Local HuggingFace
        if is_in_hf:
            try:
                print("Step 2: Checking local Hugging Face cache...")
                model_path = self.query_huggingface_model(model_name, local_only=True)
                if model_path:
                    print(f"Success: Found in local Hugging Face cache: {model_path}")
                    return self._ensure_trailing_slash(model_path)
            except FileNotFoundError:
                print("Info: Not found in local Hugging Face cache.")
            except ImportError:
                print(
                    self._logger(
                        "Warning: Hugging Face library not available for local check."
                    )
                )
            except Exception as e:
                print(
                    self._logger(
                        f"Warning: Error checking local Hugging Face cache: {e}"
                    )
                )
        else:
            print(
                "Step 2: Skipping local Hugging Face check (not in HuggingFace card)."
            )

        # 3. Download from HuggingFace
        print("Info: Model not found in local caches. Attempting downloads...")
        if is_in_hf:
            try:
                print("Step 3: Attempting download from Hugging Face...")
                model_path = self.query_huggingface_model(model_name, local_only=False)
                if model_path:
                    print(f"Success: Downloaded from Hugging Face: {model_path}")
                    return self._ensure_trailing_slash(model_path)
            except ImportError:
                print(
                    self._logger(
                        "Warning: Hugging Face library not available, cannot download."
                    )
                )
            except Exception as e:
                print(
                    f"Info: Hugging Face download failed: {e}. Trying ModelScope next."
                )
        else:
            print("Step 3: Skipping Hugging Face download (not in HuggingFace card).")

        # 4. Download from ModelScope
        if is_in_ms:
            try:
                print("Step 4: Attempting download from ModelScope...")
                model_path = self.query_modelscope_model(model_name, local_only=False)
                if model_path:
                    print(f"Success: Downloaded from ModelScope: {model_path}")
                    return self._ensure_trailing_slash(model_path)
            except ImportError:
                print(
                    self._logger(
                        "Warning: ModelScope library not available, cannot download."
                    )
                )
            except Exception as e:
                print(self._logger(f"Error: ModelScope download failed: {e}"))
        else:
            print("Step 4: Skipping ModelScope download (not in ModelScope card).")

        if model_path is None:
            error_msg = (
                f"Failed to find or download model '{model_name}' from any source."
            )
            print(self._logger(error_msg))
            raise FileNotFoundError(error_msg)

        return self._ensure_trailing_slash(model_path)


def prior_model_check(save_dir: str = "./pretrained_models") -> None:
    """Ensure prior assets (``human_model_files``, etc.) exist under ``save_dir``."""
    human_model_path = os.path.join(save_dir, "human_model_files")
    if os.path.exists(human_model_path):
        return
    if os.path.islink(human_model_path):
        try:
            os.unlink(human_model_path)
            print("Removed broken symlink: human_model_files")
        except OSError as e:
            print(f"Failed to remove broken symlink: {e}")
    print("Prior models not found or invalid. Downloading...")
    auto_query = AutoModelQuery(save_dir=save_dir)
    auto_query.download_all_prior_models()
    print("Prior models ready.")


if __name__ == "__main__":
    test_save_dir = "./pretrained_models"
    print(
        f"Initializing AutoModelQuery with save_dir: {os.path.abspath(test_save_dir)}"
    )
    automodel = AutoModelQuery(save_dir=test_save_dir)

    # Download all prior models: python -m core.utils.model_download_utils --prior
    if "--prior" in sys.argv:
        prior_names = set(HuggingFace_Prior_MODEL_CARD) | set(
            ModelScope_Prior_MODEL_CARD
        )
        print(f"\n--- Downloading prior models: {prior_names} ---")
        result = automodel.download_all_prior_models()
        for name, path in result.items():
            print(f"  {name}: {path}")
        print("Prior models download complete.")
        sys.exit(0)

    test_models = ["LHMPP-700M"]
    for model_to_test in test_models:
        print(f"\n--- Testing {model_to_test} ---")
        try:
            model_path_test = automodel.query(model_to_test)
            print(f"===> Final path for {model_to_test}: {model_path_test}")
            print(f"     Does path exist? {os.path.exists(model_path_test)}")
        except Exception as e:
            print(f"Error: {e}")
