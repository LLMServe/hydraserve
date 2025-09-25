add_library(st_pybinding SHARED pybinding.cc main.cc)
# Set the output directory for the shared library
set_target_properties(st_pybinding PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib
)