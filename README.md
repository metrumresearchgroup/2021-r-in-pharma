# 2021 R in Pharma workshop material

## Julia environment

- Navigate to the root `2021-r-in-pharma` directory.
- Start a Julia REPL (in vscode you can do `Cmd`+`Shift`+`P` -> `Julia REPL`).
- Instantiate Julia environment by typing in the REPL:

```
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```