ADD_EXECUTABLE(test_add_op test_add_op.cpp)

TARGET_LINK_LIBRARIES(test_add_op
  util
  tensor
  operator 
  ${SYSLIB}
)
