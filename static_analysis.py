import concurrent.futures
import time
import glob
import tarfile
import json
import ast
import re
import gzip
from collections import Counter

import jupytext
import pycparser

c_parser = pycparser.c_parser.CParser()
c_directive = re.compile("\s*#.*")
c_include = re.compile("\s*#include [<\"](.*)[>\"]")
cuda_bracket = re.compile("<<<.*>>>")

cpp_suffixes = set(["c", "h", "c++", "cxx", "hxx", "cpp", "hpp", "hp", "cu", "cuh", "cp", "hh", "cc"])

others = {
    "pyx": "Cython",
    "pxi": "Cython",
    "f": "Fortran",
    "for": "Fortran",
    "f90": "Fortran",
    "jl": "Julia",
    "rs": "Rust",
    "r": "R",
    "rdata": "R",
    "rmd": "R",
    "abap": "ABAP",
    "mat": "MATLAB",
    "asv": "MATLAB",
    "m": "Mathematica",
    "wl": "Mathematica",
    "nb": "Mathematica",
    "go": "Go",
    "ada": "Ada",
    "java": "Java",
    "scala": "Scala",
    "groovy": "Groovy",
    "kt": "Kotlin",
    "kts": "Kotlin",
    "cs": "C#",
    "fs": "F#",
    "swift": "Swift",
    "perl": "Perl",
    "pl": "Perl",
    "pm": "Perl",
    "rb": "Ruby",
    "hs": "Haskell",
    "lhs": "Haskell",
    "cbl": "COBOL",
    "cob": "COBOL",
}


jit_functions = set(
    ["numba.jit", "numba.njit", "numba.generated_jit", "numba.vectorize", "numba.guvectorize", "numba.cfunc"]
)

class APIVisitor(ast.NodeVisitor):
    def __init__(self, numba_imports):
        self.numba_imports = numba_imports
        self.all_imports = Counter()
        self.all_numba_references = []

    def visit_Import(self, node):
        for subnode in node.names:
            self.all_imports[subnode.name.split(".")[0]] += 1

    def visit_ImportFrom(self, node):
        if node.level == 0:
            self.all_imports[node.module.split(".")[0]] += 1

    def visit_FunctionDef(self, node):
        for x in node.decorator_list:
            n = len(self.all_numba_references)
            self.visit(x)
            if n != len(self.all_numba_references):
                self.all_numba_references[-1] = "@" + self.all_numba_references[-1]
        self.visit(node.args)
        if node.returns is not None:
            self.visit(node.returns)
        for x in node.body:
            self.visit(x)

    def visit_Call(self, node):
        n = len(self.all_numba_references)
        self.visit(node.func)
        if n != len(self.all_numba_references):
            if self.all_numba_references[-1] in jit_functions:
                unparsed = ast.unparse(node)
                self.all_numba_references[-1] = self.all_numba_references[-1] + unparsed[unparsed.index("("):]
        for x in node.args:
            self.visit(x)
        for x in node.keywords:
            self.visit(x)

    def visit_Attribute(self, node):
        if isinstance(node.ctx, ast.Load):
            name = ()
            while isinstance(node, ast.Attribute):
                name = (node.attr,) + name
                node = node.value
            if isinstance(node, ast.Name):
                name = (node.id,) + name
                self._check(".".join(name))

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self._check(node.id)

    def _check(self, name):
        for real, alias in self.numba_imports:
            if name == alias or name.startswith(alias + "."):
                fixed = real + name[len(alias):]
                self.all_numba_references.append(fixed)
                break


def analyze_python(syntax_tree):
    assert isinstance(syntax_tree, ast.Module)

    top_imports = Counter()
    numba_imports = []
    for node in syntax_tree.body:
        if isinstance(node, ast.Import):
            for subnode in node.names:
                name = subnode.name.split(".")[0]
                top_imports[name] += 1
                if name == "numba":
                    asname = subnode.name if subnode.asname is None else subnode.asname
                    numba_imports.append((subnode.name, asname))

        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            name = node.module.split(".")[0]
            top_imports[name] += 1
            if name == "numba":
                for subname in node.names:
                    asname = subname.name if subname.asname is None else subname.asname
                    numba_imports.append((node.module + "." + subname.name, asname))

    visitor = APIVisitor(numba_imports)
    try:
        visitor.visit(syntax_tree)
    except RecursionError:
        return None

    nested_imports = {k: v - top_imports.get(k, 0) for k, v in visitor.all_imports.items()}
    nested_imports = {k: v for k, v in nested_imports.items() if v != 0}

    all_numba_references = Counter()
    for x in visitor.all_numba_references:
        all_numba_references[x] += 1

    return {"top": dict(top_imports), "nested": nested_imports, "numba": dict(all_numba_references)}


def analyze_repo(filename):
    reponame = filename[49:-4]
    print(reponame)

    try:
        repodata = {"name": reponame, "python": [], "c": [], "other_language": Counter()}

        with tarfile.open(filename) as file:
            subfilenames = file.getnames()
            repodata["num_files"] = len(subfilenames)
            names_in_repo = set(x.split("/")[-1] for x in subfilenames)

            for subfilename in subfilenames:
                assert subfilename[5 : 5 + len(reponame)] == reponame
                pieces = subfilename.lower().rsplit(".", 1)
                if len(pieces) != 2 or pieces[0] == "":
                    continue

                if pieces[1] in ("py", "pyi"):
                    try:
                        subfile = file.extractfile(subfilename)
                    except (KeyError, RecursionError):
                        continue
                    if subfile is None:
                        continue

                    try:
                        syntax_tree = ast.parse(subfile.read())
                        parsable = True
                    except:
                        parsable = False

                    repodata["python"].append({
                        "name": subfilename[5 + len(reponame) + 1:],
                        "suffix": pieces[1],
                        "data": analyze_python(syntax_tree) if parsable else None,
                    })

                elif pieces[1] == "ipynb":
                    try:
                        subfile = file.extractfile(subfilename)
                    except (KeyError, RecursionError):
                        continue
                    if subfile is None:
                        continue

                    try:
                        notebook = jupytext.reads(
                            subfile.read().decode("utf-8", errors="surrogateescape")
                        )
                        text = jupytext.writes(notebook, fmt="py:percent")
                        syntax_tree = ast.parse(text)
                        parsable = True
                    except:
                        parsable = False

                    repodata["python"].append({
                        "name": subfilename[5 + len(reponame) + 1:],
                        "suffix": pieces[1],
                        "data": analyze_python(syntax_tree) if parsable else None,
                    })

                elif pieces[1] in cpp_suffixes:
                    try:
                        subfile = file.extractfile(subfilename)
                    except (KeyError, RecursionError):
                        continue
                    if subfile is None:
                        continue

                    text = subfile.read().decode("utf-8", errors="surrogateescape")

                    global_include = Counter()
                    local_include = Counter()
                    for include in c_include.finditer(text):
                        if include.group(1).split("/")[-1] in names_in_repo:
                            local_include[include.group(1)] += 1
                        else:
                            global_include[include.group(1)] += 1

                    try:
                        c_parser.parse(c_directive.sub("", text), subfilename)
                        is_c = True
                    except pycparser.plyparser.ParseError:
                        is_c = False

                    num_cuda_brackets = len(cuda_bracket.findall(text))

                    repodata["c"].append({
                        "name": subfilename[5 + len(reponame) + 1:],
                        "suffix": pieces[1],
                        "data": {
                            "global": dict(global_include),
                            "local": dict(local_include),
                            "is_c": is_c,
                            "num_cuda": num_cuda_brackets,
                        },
                    })

                else:
                    language = others.get(pieces[1])
                    if language is not None:
                        repodata["other_language"][language] += 1

        repodata["other_language"] = dict(repodata["other_language"])

        return filename + "\n", json.dumps(repodata, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n"

    except Exception as err:
        print(f"{reponame}: {type(err).__name__} {err}")
        return None, f"{reponame}: {type(err).__name__} {err}\n"


filenames = glob.glob("GitHub-numba-user-nonfork-raw-data-1Mcut-imports/*/*.tgz")

try:
    with open("input-skip.txt") as names:
        for name in names:
            filenames.remove(name.rstrip("\n"))
except FileNotFoundError:
    pass

with open("output-names.txt", "w") as names, open("output-errors.txt", "w") as errors, open("output-results.jsons", "w") as results:
    with concurrent.futures.ProcessPoolExecutor() as pool:
        start = time.time()
        for i, (name, result) in enumerate(pool.map(analyze_repo, filenames, chunksize=1)):
            if name is None:
                errors.write(result)
                errors.flush()
            else:
                results.write(result)
                names.write(name)
                now = int(time.time() - start)
                print(
                    f"{now // 3600}:{(now % 3600) // 60:02d}:{now % 60:02d}: {i+1}/{len(filenames)} = {(i+1)/len(filenames)}",
                    flush=True,
                )