# Tests are host-only executables with hand-written assertions, mirroring src/.
# No gtest: the project ships no dependency it does not need, and a test that
# needs a throwaway container to build is a test nobody runs.
#
# Every test needs a checkpoint; set ORDERIT_MODEL or pass one on the command
# line. A missing model is a skip, and a skip is never a pass.

add_executable(reference_test reference_test.cpp)
target_include_directories(reference_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME reference COMMAND reference_test)
set_tests_properties(reference PROPERTIES LABELS "cpu;f1")

add_executable(model_test model_test.cpp
  ${CMAKE_SOURCE_DIR}/src/config.cpp
  ${CMAKE_SOURCE_DIR}/src/model.cpp
  ${CMAKE_SOURCE_DIR}/src/weights.cpp
  ${CMAKE_SOURCE_DIR}/src/tokenizer.cpp
  ${CMAKE_SOURCE_DIR}/src/embed.cpp
)
target_include_directories(model_test PRIVATE ${CMAKE_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME model COMMAND model_test "${ORDERIT_MODEL}")
set_tests_properties(model PROPERTIES
  LABELS "cpu;f2;f3;slow"
  SKIP_RETURN_CODE 77
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)