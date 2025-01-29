# --------------------------------------------------------------------------
#
# Copyright (C) 2025 by Pablo Antolin
#
# This file is part of the QUGaR library.
#
# SPDX-License-Identifier:    MIT
#
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# This file was adapted from the DOLFINx project (www.fenicsproject.org)
# Copyright is reproduced below. See the license in ThirdPartyNotices.txt
# --------------------------------------------------------------------------

# Copyright (C) 2017-2023 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later

import pathlib
import shutil
import sys


def process(input_dir, output_dir):
    """Convert C++ demos in the Jupytext 'light' format into MyST
    flavoured markdown (and ipynb?) using Jupytext. These files can then be
    included in Sphinx documentation.

    """

    import jupytext

    # Make demo doc directory
    # demo_doc_dir = pathlib.Path("./demos")
    demo_doc_dir = pathlib.Path(output_dir)
    demo_doc_dir.mkdir(parents=True, exist_ok=True)

    # Directories to scan demo code files
    # demo_dirs = pathlib.Path("../src/demos")
    demo_dirs = pathlib.Path(input_dir)

    # Iterate over subdirectories containing demos
    for demo_subdir in demo_dirs.iterdir():
        if demo_subdir.is_dir():
            fname = pathlib.Path("/demo_" + demo_subdir.name)
            demo_doc_subdir = demo_doc_dir / fname.name
            demo_doc_subdir.mkdir(parents=True, exist_ok=True)
            # Process each demo using jupytext/myst
            for demo_file in demo_subdir.glob("main.cpp"):
                # # Copy demo files into documentation demo directory
                # shutil.copy(demo_file, demo_doc_subdir)
                cpp_demo = jupytext.read(demo_file)
                cpp_myst_text = jupytext.writes(cpp_demo, fmt="myst")

            # myst-parser does not process blocks with {code-cell}
            cpp_myst_text = cpp_myst_text.replace("{code-cell}", "cpp")

            # Similarly for python file, dump python myst text in cpp myst
            for pydemo_file in demo_subdir.glob("*.py"):
                shutil.copy(pydemo_file, demo_doc_subdir)
                python_demo = jupytext.read(pydemo_file)
                python_myst_text = jupytext.writes(python_demo, fmt="myst")
                python_myst_text = python_myst_text.replace("{code-cell}", "python")
                cpp_myst_text = cpp_myst_text.replace("![ufl-code]", python_myst_text)

            # for cmake_file in demo_subdir.glob("CMakeLists.txt"):
            #     shutil.copy(cmake_file, demo_doc_subdir)

            myst_file = (demo_doc_dir / fname.name).with_suffix(".md")
            with open(myst_file, "w") as fw:
                fw.write(f"# Demo {demo_subdir.name}\n")
                fw.write(cpp_myst_text)

            # There is a posibility to use jupyter-notebooks with C++/C kernels
            # ipynb_file = (demo_doc_dir / fname.name).with_suffix(".ipynb")
            # jupytext.write(cpp_demo, ipynb_file, fmt="ipynb")


if __name__ == "__main__":
    try:
        import jupytext

        input_dir = sys.argv[1]
        output_dir = sys.argv[2]
        process(input_dir, output_dir)
    except ImportError:
        pass  # module doesn't exist, deal with it.
