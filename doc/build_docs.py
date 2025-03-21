# Script adapted from
# https://www.codingwiththomas.com/blog/my-sphinx-best-practice-for-a-multiversion-documentation-in-different-languages
# https://github.com/ThoSe1990/SphinxExample

import os
import shutil
import subprocess

import qugar
from utils import get_git_branch

# Getting the current branch
branch = get_git_branch()
version = qugar.__version__ if branch == "release" else branch


# Creating needed directories
shutil.rmtree("./pages", ignore_errors=True)
shutil.rmtree("./generated", ignore_errors=True)
shutil.rmtree("./build", ignore_errors=True)

os.makedirs("generated", exist_ok=True)
os.makedirs("build", exist_ok=True)
os.makedirs("build/doxygen_xml", exist_ok=True)
os.makedirs("build/doxygen_html/doxygen_html", exist_ok=True)
os.makedirs("build/sphinx_html", exist_ok=True)


# Running Doxygen
try:
    subprocess.run("doxygen Doxyfile", shell=True, check=True)
except subprocess.CalledProcessError:
    raise Exception("Doxygen failed.")

# Running sphinx
try:
    subprocess.run(
        "python3 -m sphinx -b html ./ build/sphinx_html",
        shell=True,
        check=True,
    )
except subprocess.CalledProcessError:
    raise Exception("Sphinx failed.")

# Moving created documentation to pages for being uploaded.
# output_dir = "./pages/" + version
output_dir = "./pages"
os.makedirs(output_dir, exist_ok=True)
subprocess.run("mv ./build/sphinx_html/* " + output_dir + "/", shell=True)

# Cleaning directories
shutil.rmtree("./build", ignore_errors=True)
shutil.rmtree("./generated", ignore_errors=True)
