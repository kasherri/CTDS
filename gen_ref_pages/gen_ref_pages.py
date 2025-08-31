"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

# Only generate the main ctds module reference
with mkdocs_gen_files.open("reference/ctds.md", "w") as fd:
    fd.write("::: ctds")

# Create a simple navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.write("* [CTDS](ctds.md)\n")
