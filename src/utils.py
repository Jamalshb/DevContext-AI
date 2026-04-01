from urllib.parse import urlparse


def extract_repo_name(repo_url: str) -> str:
    path = urlparse(repo_url).path.strip("/")
    repo_name = path.split("/")[-1]

    if repo_name.endswith(".git"):
        repo_name = repo_name[:-4]

    return repo_name.lower()