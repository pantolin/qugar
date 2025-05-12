# Copyright (C) 2017-2023 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import os
import pathlib
import runpy
import shutil

import jupytext


def process_python_demos(input_dir: str, output_dir: str):
    """Convert Python demos in the Jupytext 'light' format into MyST
    flavoured markdown and ipynb using Jupytext. These files can then be
    included in Sphinx documentation.

    """
    # Directories to scan
    subdirs = [pathlib.Path(input_dir)]

    # Make demo doc directory
    demo_dir = pathlib.Path(output_dir)
    demo_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over subdirectories containing demos
    for subdir in subdirs:
        # Process each demo using jupytext/myst
        for demo in subdir.glob("**/demo*.py"):
            python_demo = jupytext.read(demo)
            myst_text = jupytext.writes(python_demo, fmt="myst")

            # myst-parser does not process blocks with {code-cell}
            myst_text = myst_text.replace("{code-cell}", "python")
            myst_file = (demo_dir / demo.name).with_suffix(".md")
            with open(myst_file, "w") as fw:
                fw.write(myst_text)

            ipynb_file = (demo_dir / demo.name).with_suffix(".ipynb")
            jupytext.write(python_demo, ipynb_file, fmt="ipynb")

            # Copy python demo files into documentation demo directory
            shutil.copy(demo, demo_dir)

            # If demo saves matplotlib images, run the demo to create
            # images
            code = demo.read_text()
            if "savefig" in code:
                demo = demo.resolve()
                here = os.getcwd()
                os.chdir(demo_dir)
                runpy.run_path(str(demo))
                os.chdir(here)

        # Copy images used in demos from the assets directory
        assets_dir = subdir / "assets"
        if assets_dir.exists() and assets_dir.is_dir():
            # demo_assets is the target directory: output_dir/assets
            # output_dir is typically "generated/demos"
            # so demo_assets becomes "generated/demos/assets"
            target_demo_assets_dir = demo_dir / "assets"
            target_demo_assets_dir.mkdir(parents=True, exist_ok=True)
            # Copy all png and pdf files from the source assets directory
            for file in assets_dir.glob("**/*.png"):
                shutil.copy(file, target_demo_assets_dir / file.name)
            for file in assets_dir.glob("**/*.pdf"):
                shutil.copy(file, target_demo_assets_dir / file.name)


if __name__ == "__main__":
    process_python_demos(input_dir="../python/demo", output_dir="generated/demos")
