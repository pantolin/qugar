# Script adapted from
# https://www.codingwiththomas.com/blog/my-sphinx-best-practice-for-a-multiversion-documentation-in-different-languages
# https://github.com/ThoSe1990/SphinxExample

import os
import shutil
import subprocess

import yaml


def build_doc(version, tag, main_branch):
    shutil.rmtree("./generated", ignore_errors=True)
    shutil.rmtree("./build", ignore_errors=True)

    os.makedirs("generated", exist_ok=True)
    os.makedirs("build", exist_ok=True)
    os.makedirs("build/doxygen_xml", exist_ok=True)
    os.makedirs("build/doxygen_html/doxygen_html", exist_ok=True)
    os.makedirs("build/sphinx_html", exist_ok=True)

    os.environ["current_version"] = version

    subprocess.run("git checkout " + tag, shell=True)
    subprocess.run(f"git checkout {main_branch} -- conf.py", shell=True)
    subprocess.run(f"git checkout {main_branch} -- versions.yaml", shell=True)
    subprocess.run(f"git checkout {main_branch} -- Doxyfile", shell=True)
    subprocess.run(f"git checkout {main_branch} -- jupytext_process.py", shell=True)

    try:
        subprocess.run("doxygen Doxyfile", shell=True, check=True)
    except subprocess.CalledProcessError:
        raise Exception("Doxygen failed.")

    try:
        subprocess.run(
            "python3 -m sphinx -b html ./ build/sphinx_html",
            shell=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        raise Exception("Sphinx failed.")

    output_dir = "./pages/" + version
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run("mv ./build/sphinx_html/* " + output_dir + "/", shell=True)

    shutil.rmtree("./build", ignore_errors=True)
    shutil.rmtree("./generated", ignore_errors=True)


os.environ["build_all_docs"] = str(True)
os.environ["pages_root"] = "https://pantolin.github.io/qugar"

main_branch = "main"

shutil.rmtree("./pages", ignore_errors=True)

build_doc("latest", main_branch, main_branch)

with open("versions.yaml", "r") as yaml_file:
    docs = yaml.safe_load(yaml_file)

if docs is not None:
    for version, details in docs.items():
        tag = details.get("tag", "")
        build_doc(version, tag, main_branch)
