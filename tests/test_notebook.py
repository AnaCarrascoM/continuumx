"""The example notebook, run as written.

The notebook ``examples/emergency_network.ipynb`` is stored with its outputs,
so that it can be read on GitHub without running it.  This module executes
its code cells in order and checks that every stored output is still the one
the package produces, so that the notebook and the package cannot drift apart.
"""

import ast
import json
from pathlib import Path

NOTEBOOK = (
    Path(__file__).resolve().parent.parent / "examples" / "emergency_network.ipynb"
)


def _run(source, namespace):
    """Execute a cell and return the repr of its last expression, if any."""
    tree = ast.parse(source)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, "<cell>", "exec"), namespace)
    if last is None:
        return None
    return repr(eval(compile(last, "<cell>", "eval"), namespace))


def test_notebook_outputs_are_up_to_date():
    cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    namespace: dict = {}
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        stored = [
            "".join(out["data"]["text/plain"])
            for out in cell["outputs"]
            if out["output_type"] == "execute_result"
        ]
        assert _run(source, namespace) == (stored[0] if stored else None), source
