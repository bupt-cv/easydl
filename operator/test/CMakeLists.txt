ADD_EXECUTABLE(test_add_op test_add_op.cpp)
TARGET_LINK_LIBRARIES(test_add_op
  util
  tensor
  operator 
  ${SYSLIB}
)

ADD_EXECUTABLE(test_operator test_operator.cpp)
TARGET_LINK_LIBRARIES(test_operator
  -Wl,--whole-archive
  util
  tensor
  operator 
  -Wl,--no-whole-archive
  ${SYSLIB}
)
