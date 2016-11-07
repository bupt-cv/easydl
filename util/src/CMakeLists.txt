aux_source_directory(./ SRCS)

file(GLOB CUSRCS *.cu)

# static library
CUDA_ADD_LIBRARY(util SHARED ${SRCS} ${CUSRCS})
