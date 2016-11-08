aux_source_directory(./ SRCS)

ADD_LIBRARY(easydl SHARED ${SRCS})
SET_TARGET_PROPERTIES(easydl PROPERTIES PREFIX "")
TARGET_LINK_LIBRARIES(easydl tensor operator util ${SYSLIB})
