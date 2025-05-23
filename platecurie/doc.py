# Copyright 2019 Pascal Audet
#
# This file is part of PlateCurie.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Use the `platecurie.doc.install_doc` function to copy all
Jupyter Notebooks and example data to a local directory.

"""

import os
import shutil
import importlib.resources as resources


def install_doc(path="./PlateCurie-Examples"):
    """
    Install the examples for PlateCurie in the given location.

    WARNING: If the path exists, the files will be written into the path
    and will overwrite any existing files with which they collide. The default
    path ("./PlateCurie-Examples") is chosen to make collision less likely/problematic

    Example applications of PlateCurie are in the form of jupyter notebooks.
    """

    # Access the package resources
    # Make sure 'platecurie' is a package and 'examples' is a subdirectory/package
    with resources.path("platecurie", "examples") as notebooks_path:
        # Use shutil.copytree() to copy directory contents
        if os.path.exists(path):
            raise FileExistsError(f"The target directory {path} already exists.")
        shutil.copytree(
            notebooks_path,
            path,
            copy_function=shutil.copy2,
        )
