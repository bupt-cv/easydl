aux_source_directory(./ SRCS)

# static library
ADD_LIBRARY(operator STATIC ${SRCS})
TARGET_LINK_LIBRARIES(operator util tensor ${SYSLIB})
