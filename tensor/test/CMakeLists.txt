ADD_EXECUTABLE(test_tensor test_tensor.cc)

TARGET_LINK_LIBRARIES(test_tensor
  tensor 
  ${SYSLIB}
)
