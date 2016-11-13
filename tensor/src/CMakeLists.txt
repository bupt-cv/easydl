aux_source_directory(./ SRCS)

# static library
ADD_LIBRARY(tensor STATIC ${SRCS})
