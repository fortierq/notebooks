from pathlib import Path
import subprocess
import os


def convert(files):
    dir = Path(__file__).parents[1]
    os.chdir(dir / "jekyll_export")
    for file, title in files:
        file = Path(file)
        if str(file).find("checkpoint") == -1:
            output = Path("/home/qfortier/Documents/code/fortierq.github.io/_pages/nb") / file.with_suffix(".html").name.lower()
            Path.mkdir(output.parent, parents=True, exist_ok=True)
            print(f"\nConvert notebook {file} to {output}\n")
            print(subprocess.run(f"jupyter nbconvert --execute --to exporter.JekyllExporter --template jekyll {str(file)} --output {output}", 
                shell=True,
                capture_output=True))
            with open(output, "r") as f:
                html_output = f.read()
            with open(output, "w") as f:
                i, j = html_output.find("<h1>"), html_output.find("</h1>")
                if i != -1:
                    html_output = html_output[:i] + html_output[j + 5:]
                url = file.stem.lower().replace(' ', '')
                toc = "false" if title is None else "true"
                f.write(f"---\npermalink: /nb/{url}/\nlayout: nb\nauthor_profile: false\ntoc: {toc}\ntoc_label: {title} \ntoc_sticky: true\n---\n\n")
                f.write(html_output.replace("&#182;", ''))

files = [
    ("/home/qfortier/Documents/code/ML/ML/SVM.ipynb", None),
    ("/home/qfortier/Documents/code/ML/ML/regression_lineaire.ipynb", "Régression linéaire"),
    # "/home/qfortier/Documents/code/ML/ML/KMeans.ipynb",
    ("/home/qfortier/Documents/code/ML/ML/logistic.ipynb", "Régression logistique"),
    ("/home/qfortier/Documents/code/ML/image_processing/hist_equal.ipynb", "Égalisation d'histogramme")
]
convert(files)