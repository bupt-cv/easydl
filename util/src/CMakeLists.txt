aux_source_directory(./ SRCS)

# static library
ADD_LIBRARY(util STATIC ${SRCS})

