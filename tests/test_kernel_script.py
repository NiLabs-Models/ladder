"""Static checks on the Kaggle kernel script.

The kernel only ever runs on Kaggle, so nothing here can execute it. But its
first version shipped a bug that cost a whole GPU session: stages were written
as `@stage("setup")` decorators that ran the function at decoration time and
rebound the name to the *result*, and the next line then called that result --
`TypeError: 'NoneType' object is not callable`, after the pip install had
already failed. These checks are cheap and catch that class of mistake.
"""

import ast
from pathlib import Path

KERNEL = Path(__file__).resolve().parents[1] / "kaggle" / "ladder_kernel.py"


def parsed():
    return ast.parse(KERNEL.read_text(encoding="utf-8"))


def test_kernel_script_parses():
    assert parsed()


def test_stage_results_are_never_called():
    tree = parsed()

    # Names bound to the result of run_stage(...).
    stage_results = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            if isinstance(fn, ast.Name) and fn.id == "run_stage":
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        stage_results.add(target.id)

    assert stage_results, "expected the kernel to capture some run_stage results"

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    clashes = stage_results & called
    assert not clashes, f"stage results called as functions: {sorted(clashes)}"


def test_no_stage_decorators_remain():
    for node in ast.walk(parsed()):
        if isinstance(node, ast.FunctionDef):
            for dec in node.decorator_list:
                name = dec.func.id if isinstance(dec, ast.Call) else getattr(dec, "id", "")
                assert name != "stage", f"{node.name} still uses the @stage decorator"


def test_preflight_runs_before_the_heavy_install():
    """A missing entitlement should fail in seconds, not inside a pip timeout."""
    tree = parsed()
    order = [
        node.value.args[0].value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_stage"
        and isinstance(node.value.args[0], ast.Constant)
    ]
    order += [
        node.value.args[0].value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "run_stage"
        and isinstance(node.value.args[0], ast.Constant)
    ]
    assert "preflight" in order
    assert order.index("preflight") < order.index("setup")


def test_base_eval_is_scheduled_before_training():
    """A tuned number with nothing to compare it against is not a result."""
    source = KERNEL.read_text(encoding="utf-8")
    assert source.index('run_stage("eval_base"') < source.index('run_stage("train"')
