from pathlib import Path
import subprocess
import os
import re
import base64
            
dir = Path(__file__).parents[1]

def convert(files):
    os.chdir(dir / "jekyll_export")
    re_img = re.compile('<img src="(.*\.png)"')
    re_title = re.compile('<h1.*>(.*)<a.*</h1>')
    for file in files:
        file = Path(file).resolve()
        title = "Sommaire"
        def repl_title(match):
            nonlocal title
            title = match.group(1)
            print(f"Title: {title}")
            return ''
        def repl_img(match):
            p = (file.parent / match.group(1)).resolve()
            print(f"Convert image {p}\n")
            if p.is_file():
                with open(p, "rb") as f:
                    s = base64.b64encode(f.read()).decode('ascii')
                    return f'<img src="data:image/png;base64,{s}"'

        if str(file).find("checkpoint") == -1:
            output = Path("/home/qfortier/Documents/code/fortierq.github.io/_pages/nb") / file.with_suffix(".html").name.lower()
            Path.mkdir(output.parent, parents=True, exist_ok=True)
            print(f"\nConvert notebook {file} to {output}\n")
            print(subprocess.run(f"poetry run jupyter nbconvert --execute --to exporter.JekyllExporter --template jekyll {str(file)} --output {output}", 
                shell=True,
                capture_output=True))
            with open(output, "r") as f:
                html_output = f.read()
                i, j = html_output.find("<h1>"), html_output.find("</h1>")
                if i != -1:
                    html_output = html_output[:i] + html_output[j + 5:]
                html_output = re_img.sub(repl_img, html_output)
                html_output = re_title.sub(repl_title, html_output, count=1)
            with open(output, "w") as f:
                url = file.stem.lower().replace(' ', '')
                toc = "false" if title is None else "true"
                f.write(f"---\npermalink: /nb/{url}/\nlayout: nb\nauthor_profile: false\ntoc: {toc}\ntoc_label: {title} \ntoc_sticky: true\n---\n\n")
                f.write(html_output.replace("&#182;", ''))


files = [
    dir / "ML/SVM.ipynb",
    dir / "ML/regression_lineaire.ipynb",
    # dir / "ML/KMeans.ipynb",
    dir / "ML/logistic.ipynb",
    dir / "image_processing/hist_equal.ipynb",
    dir / "optimisation/pavage.ipynb", 
]
convert(files)