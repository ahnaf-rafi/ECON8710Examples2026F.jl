You can generate a Julia project as follows.
In the parent directory of the project, open a `julia` repl and run the following.
```
]
generate MyProject
```
This will create a `MyProject` directory with a `MyProject/Project.toml` and a
`MyProject/src/MyProject.jl`.

If you have already created the project directory, then open a `julia` repl in
project directory and run the following.
```julia
]
activate .
```

Add packages with:
```julia
pkg> add LinearAlgebra Statistics Random Distributions Optim
```

If a Project.toml and Manifest.toml already exist (e.g. you did `generate
MyProject`), then these packages will get written to those files. Otherwise,
the first run of `pkg> add` will create these files and add the package list to
them.
