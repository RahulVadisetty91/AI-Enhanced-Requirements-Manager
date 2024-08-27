import glob
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from itertools import chain
from os.path import dirname, isfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from packaging.requirements import Requirement
from packaging.version import Version
from sklearn.ensemble import RandomForestClassifier  # Example for AI feature integration
from sklearn.preprocessing import LabelEncoder

REQUIREMENT_FILES = {
    "pytorch": (
        "requirements/pytorch/base.txt",
        "requirements/pytorch/extra.txt",
        "requirements/pytorch/strategies.txt",
        "requirements/pytorch/examples.txt",
    ),
    "fabric": (
        "requirements/fabric/base.txt",
        "requirements/fabric/strategies.txt",
    ),
    "data": ("requirements/data/data.txt",),
}
REQUIREMENT_FILES_ALL = list(chain(*REQUIREMENT_FILES.values()))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class _RequirementWithComment(Requirement):
    strict_string = "# strict"

    def __init__(self, *args: Any, comment: str = "", pip_argument: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        assert pip_argument is None or pip_argument  # sanity check that it's not an empty str
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: str) -> str:
        """Remove version restrictions unless they are strict."""
        out = str(self)
        if self.strict:
            return f"{out}  {self.strict_string}"
        specs = [(spec.operator, spec.version) for spec in self.specifier]
        if unfreeze == "major":
            for operator, version in specs:
                if operator in ("<", "<="):
                    major = Version(version).major
                    # replace upper bound with major version increased by one
                    return out.replace(f"{operator}{version}", f"<{major + 1}.0")
        elif unfreeze == "all":
            for operator, version in specs:
                if operator in ("<", "<="):
                    # drop upper bound
                    return out.replace(f"{operator}{version},", "")
        elif unfreeze != "none":
            raise ValueError(f"Unexpected unfreeze: {unfreeze!r} value.")
        return out


def _parse_requirements(lines: Iterable[str]) -> Iterator[_RequirementWithComment]:
    """Adapted from `pkg_resources.parse_requirements` to include comments."""
    pip_argument = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " #" in line:
            comment_pos = line.find(" #")
            line, comment = line[:comment_pos], line[comment_pos:]
        else:
            comment = ""
        if line.startswith("--"):
            pip_argument = line
            continue
        if line.startswith("-r "):
            continue
        yield _RequirementWithComment(line, comment=comment, pip_argument=pip_argument)
        pip_argument = None


def load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: str = "all") -> List[str]:
    """Loading requirements from a file."""
    assert unfreeze in {"none", "major", "all"}
    path = Path(path_dir) / file_name
    if not path.exists():
        logging.warning(f"Folder {path_dir} does not have any base requirements.")
        return []
    assert path.exists(), (path_dir, file_name, path)
    text = path.read_text().splitlines()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]


def load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as description."""
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fo:
        text = fo.read()

    # drop images from readme
    text = text.replace(
        "![PT to PL](docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif)", ""
    )

    github_source_url = os.path.join(homepage, "raw", version)
    text = text.replace(
        "docs/source-pytorch/_static/", f"{os.path.join(github_source_url, 'docs/source-app/_static/')}"
    )

    text = text.replace("badge/?version=stable", f"badge/?version={version}")
    text = text.replace("pytorch-lightning.readthedocs.io/en/stable/", f"pytorch-lightning.readthedocs.io/en/{version}")
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{version}/graph/badge.svg")
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={version}")
    text = text.replace("?branchName=master", f"?branchName=refs%2Ftags%2F{version}")

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    return re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)


def distribute_version(src_folder: str, ver_file: str = "version.info") -> None:
    """Copy the global version to all packages."""
    ls_ver = glob.glob(os.path.join(src_folder, "*", "__version__.py"))
    ver_template = os.path.join(src_folder, ver_file)
    for fpath in ls_ver:
        fpath = os.path.join(os.path.dirname(fpath), ver_file)
        print("Distributing the version to", fpath)
        if os.path.isfile(fpath):
            os.remove(fpath)
        shutil.copy2(ver_template, fpath)


def _load_aggregate_requirements(req_dir: str = "requirements", freeze_requirements: bool = False) -> None:
    """Load all base requirements from all particular packages and prune duplicates."""
    requires = [
        load_requirements(d, unfreeze="none" if freeze_requirements else "major")
        for d in glob.glob(os.path.join(req_dir, "*"))
        if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*"))) > 0 and not os.path.basename(d).startswith("_")
    ]
    if not requires:
        return
    requires = sorted(set(chain(*requires)))
    with open(os.path.join(req_dir, "base.txt"), "w") as fp:
        fp.writelines([ln + os.linesep for ln in requires] + [os.linesep])


def _retrieve_files(directory: str, *ext: str) -> List[str]:
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(os.path.join(root, fname))
    return all_files


def _replace_imports(lines: List[str], mapping: List[Tuple[str, str]], lightning_by: str = "") -> List[str]:
    """Replace imports of standalone package to lightning."""
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            out[i] = re.sub(
                rf"([^_/@]|^){source_import}([^_\w/]|$)",
                rf"\1{target_import}\2",
                ln,
            )
            if lightning_by:
                out[i] = out[i].replace("from lightning import ", f"from {lightning_by} import ")
                out[i] = out[i].replace("import lightning ", f"import {lightning_by} ")
    return out


def copy_replace_imports(
    source_dir: str,
    source_imports: Sequence[str],
    target_imports: Sequence[str],
    target_dir: Optional[str] = None,
    lightning_by: str = "",
) -> None:
    """Copy package content with import adjustments."""
    print(f"Replacing imports: {locals()}")
    assert len(source_imports) == len(target_imports), (
        "source and target imports must have the same length, "
        f"source: {len(source_imports)}, target: {len(target_imports)}"
    )
    if target_dir is None:
        target_dir = source_dir

    ls = _retrieve_files(source_dir)
    for fp in ls:
        fp_new = fp.replace(source_dir, target_dir)
        _, ext = os.path.splitext(fp)
        if ext in (".png", ".jpg", ".ico"):
            os.makedirs(dirname(fp_new), exist_ok=True)
            if not isfile(fp_new):
                shutil.copy(fp, fp_new)
            continue
        if ext in (".pyc",):
            continue
        with open(fp, encoding="utf-8") as fo:
            try:
                lines = fo.readlines()
            except UnicodeDecodeError as e:
                logging.error("Reading %s fails due to %s.", fp, e)
                continue
        if not lines:
            continue
        lines = _replace_imports(lines, list(zip(source_imports, target_imports)), lightning_by)
        if lines:
            os.makedirs(dirname(fp_new), exist_ok=True)
            with open(fp_new, "w", encoding="utf-8") as fo:
                fo.writelines(lines)
            print(f"Updated imports in {fp_new}")


def create_mirror_package(
    source_dir: str,
    target_dir: str,
    *args: str,
    mirror_by: str = "",
    **kwargs: Any,
) -> None:
    """Mirror package with import adjustments."""
    copy_replace_imports(source_dir, *args, **kwargs)
    distribute_version(source_dir)
    os.makedirs(target_dir, exist_ok=True)
    shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
    if mirror_by:
        with open(os.path.join(target_dir, "setup.py"), "a", encoding="utf-8") as fo:
            fo.write(f"\n# {mirror_by} mirror setup\n")


class AssistantCLI:
    """CLI Assistant for managing requirements and packages."""

    @staticmethod
    def load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: str = "all") -> List[str]:
        return load_requirements(path_dir, file_name, unfreeze)

    @staticmethod
    def copy_replace_imports(
        source_dir: str,
        source_imports: Sequence[str],
        target_imports: Sequence[str],
        target_dir: Optional[str] = None,
        lightning_by: str = "",
    ) -> None:
        copy_replace_imports(source_dir, source_imports, target_imports, target_dir, lightning_by)

    @staticmethod
    def create_mirror_package(
        source_dir: str,
        target_dir: str,
        *args: str,
        mirror_by: str = "",
        **kwargs: Any,
    ) -> None:
        create_mirror_package(source_dir, target_dir, *args, mirror_by=mirror_by, **kwargs)

    @staticmethod
    def distribute_version(src_folder: str, ver_file: str = "version.info") -> None:
        distribute_version(src_folder, ver_file)


# Example AI-enhanced features
def predict_version_adjustments(requirements: List[str]) -> List[str]:
    """Predict necessary version adjustments using a simple AI model."""
    # Example model: a RandomForestClassifier for demo purposes
    model = RandomForestClassifier()  # Replace with actual model training
    le = LabelEncoder()
    req_encoded = le.fit_transform(requirements)
    predictions = model.predict(req_encoded.reshape(-1, 1))
    return [req for req, pred in zip(requirements, predictions) if pred == 1]


def detect_anomalies(requirements: List[str]) -> List[str]:
    """Detect anomalies in requirements using simple statistics."""
    anomalies = []
    # Example: find requirements with unexpected patterns
    for req in requirements:
        if re.search(r"(>=|<=)\d+\.\d+", req):
            anomalies.append(req)
    return anomalies


def suggest_dependencies(requirements: List[str]) -> List[str]:
    """Provide AI-based suggestions for dependency adjustments."""
    suggestions = []
    # Example: Use a dummy recommendation engine
    # Replace with actual AI-based suggestions
    for req in requirements:
        if "tensorflow" in req.lower():
            suggestions.append(req.replace("tensorflow", "tensorflow-cpu"))
    return suggestions


# Example usage of AI features
if __name__ == "__main__":
    reqs = load_requirements("requirements")
    predictions = predict_version_adjustments(reqs)
    anomalies = detect_anomalies(reqs)
    suggestions = suggest_dependencies(reqs)
    print("Predicted Adjustments:", predictions)
    print("Detected Anomalies:", anomalies)
    print("Suggestions:", suggestions)
