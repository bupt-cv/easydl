ADD_EXECUTABLE(test_tensor test_tensor.cpp)

TARGET_LINK_LIBRARIES(test_tensor
  tensor 
  ${SYSLIB}
)
