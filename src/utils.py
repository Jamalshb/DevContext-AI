import re
from urllib.parse import urlparse

_GITHUB_RE = re.compile(r"^[\w.-]+/[\w.-]+$")


def normalize_repo_url(repo_url: str) -> str:
    """Accept 'https://github.com/owner/repo(.git)' or the short form 'owner/repo'."""
    url = (repo_url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Please enter a repository URL.")
    if _GITHUB_RE.match(url):
        return f"https://github.com/{url}"
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("Invalid URL. Example: https://github.com/owner/repo")
    if len([p for p in parsed.path.split("/") if p]) < 2:
        raise ValueError("The URL must contain owner and repo, e.g. https://github.com/owner/repo")
    return url


def extract_repo_name(repo_url: str) -> str:
    path = urlparse(normalize_repo_url(repo_url)).path.strip("/")
    repo_name = path.split("/")[-1]
    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]
    return repo_name.lower()


def repo_slug(repo_url: str) -> str:
    """Unique, filesystem-safe id like 'owner__repo' (used for the DB folder/collection)."""
    parts = [p for p in urlparse(normalize_repo_url(repo_url)).path.split("/") if p]
    owner, name = parts[-2], parts[-1]
    if name.endswith(".git"):
        name = name[:-4]
    slug = re.sub(r"[^a-zA-Z0-9_-]", "-", f"{owner}__{name}").lower()
    return slug[:60]
