macro(qugar_find_Python)
  find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule)
endmacro()
