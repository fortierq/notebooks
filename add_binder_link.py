import sys
import re

def add_link(repo, nb, env, branch_repo="main", branch_env="main"):
    with open(nb, 'r') as f:
        lines = f.read()
        m = re.search('"# (.*)\\\\n', lines)
        if not m:
            print(f"Error: {nb} does not have a title")
            return
        if "Binder" in m.string:
            print(f"Binder link is already there")
            return
        i, j = m.start(1), m.end(1)
        title = m.group(1)
        repo_nb = repo.split("/")[-1] + "/" + nb
        def F(s): return s.replace('/', '%252F')
        url = f"https://mybinder.org/v2/gh/{ env }/{ branch_env }?urlpath=git-pull%3Frepo%3Dhttps%253A%252F%252Fgithub.com%252F{ F(repo) }%26urlpath%3Dlab%252Ftree%252F{ F(repo_nb) }%26branch%3D{ branch_repo }"
        title = f"<center><a href='{url}'>{title} <img src=https://mybinder.org/badge_logo.svg></a></center>"

    with open(nb, 'w') as f:
        f.write(lines[:i] + title + lines[j:])

add_link(sys.argv[1], sys.argv[2], sys.argv[3], "master")

# python3 add_binder_link.py nb/optimisation/pavage/pavage.ipynb repo env