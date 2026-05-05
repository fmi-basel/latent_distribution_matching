# check_docs.py
import ast
import os
import re
from typing import List, Tuple

ARG_SECTION = re.compile(r"Args:\s*(.*?)\n\n", re.S)
RET_SECTION = re.compile(r"Returns:\s*(.*?)\n\n", re.S)

def extract_sections(doc: str) -> Tuple[List[str], List[str]]:
    """
    Extracts the Args and Returns sections from a docstring.
    Args:
        doc (str): The docstring to extract sections from.
    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists:
    """
    args = []
    rets = []
    m = ARG_SECTION.search(doc)
    if m:
        args = [line.strip().split(":")[0] for line in m.group(1).splitlines() if line.strip()]
    m = RET_SECTION.search(doc)
    if m:
        # capture something like "dict", "torch.Tensor"
        ret = m.group(1).split(":")[0].strip()
        rets = [ret]
    return args, rets

def check_file(path: str):
    """
    Check a file for docstring consistency.
    Args:
        path (str): The path to the file to check.
    """
    with open(path, "r") as f:
        tree = ast.parse(f.read(), path)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            sig_args = [arg.arg for arg in node.args.args]
            # drop self/cls
            if sig_args and sig_args[0] in ("self","cls"):
                sig_args = sig_args[1:]
            doc_args, doc_rets = extract_sections(doc)
            # compare Args
            missing = set(sig_args) - set(doc_args)
            extra = set(doc_args) - set(sig_args)
            # compare Returns only by presence if annotated
            anno = getattr(node.returns, "id", None) or getattr(node.returns, "attr", None)
            ret_mismatch = bool(anno) and anno not in doc_rets
            if missing or extra or ret_mismatch:
                print(f"{path}:{node.lineno} → {node.name}()")
                if missing:
                    print(f"  ✗ missing in doc: {missing}")
                if extra:
                    print(f"  ✗ extra in doc:   {extra}")
                if ret_mismatch:
                    print(f"  ✗ return mismatch: signature returns '{anno}' but doc returns {doc_rets}")
                print()

def main():
    for root, _, files in os.walk("."):
        if root.startswith("./.") or "venv" in root:
            continue
        for fn in files:
            if fn.endswith(".py"):
                check_file(os.path.join(root, fn))

if __name__ == "__main__":
    main()