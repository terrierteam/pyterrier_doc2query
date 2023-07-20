import shutil
import json
from hashlib import sha256
from pathlib import Path
import git

_git_has_lfs_ = None
def _git_has_lfs():
    global _git_has_lfs_
    if _git_has_lfs_ is not None:
        return _git_has_lfs_
    import git.cmd
    try:
        git.cmd.Git(".").execute(["git", "lfs", "env"])
        _git_has_lfs_ = True
    except git.exc.GitCommandError:
        _git_has_lfs_ = False
    return _git_has_lfs_

class Artefact:
    @classmethod
    def from_repo(cls, repo_url, *args, **kwargs):
        assert _git_has_lfs(), "git lfs support is required"
        repo_url_hash = sha256(repo_url.encode()).hexdigest()
        path = Path.home()/'.pyterrier'/'repos'/repo_url_hash
        meta_path = Path.home()/'.pyterrier'/'repos'/(repo_url_hash + '.json')
        if not meta_path.exists():
            repo = git.Repo.clone_from(repo_url, str(path), **{'depth': '1'})
            del repo
            shutil.rmtree(path/'.git')
            with meta_path.open('wt') as fout:
                json.dump({'repo_url': repo_url}, fout)
        return cls(path, *args, **kwargs)
