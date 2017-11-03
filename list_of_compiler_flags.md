### List of relevant compiler options

##### Compiler
```makefile
--include_path="${PROJECT_ROOT}"
--include_path="${PROJECT_ROOT}/database"
--include_path="${PROJECT_ROOT}/fann/inc"
--printf_support=full
--define=FIXEDFANN # to enable usage of fixed-points
```

##### Linker
```makefile
--heap_size=1500 # for dynamic memory allocation
```