import shutil
import json
from hashlib import sha256
from pathlib import Path
import git

class Artefact:
    @classmethod
    def from_repo(cls, repo_url, *args, **kwargs):
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
