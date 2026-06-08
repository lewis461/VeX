# VeX Plugin - semantic-aware sharded HNSW indices for host + DPU vector search.
# Added to the FAISS build via `add_subdirectory(plugin)` (see build.sh).

set(PLUGIN_SRCS
  IndexShardBase.cpp
  IndexKmeansShard.cpp
  IndexKmeansShardFinegrained.cpp   # VeX: fine-grained k-means + PCA/greedy merge
  IndexRandomlyShard.cpp
)

set(PLUGIN_HEADERS
  IndexShardBase.h
  IndexKmeansShard.h
  IndexKmeansShardFinegrained.h
  IndexRandomlyShard.h
)

# DOCA (BlueField DPU) sources are only needed for the DPU/pipelined search path.
# Host-only builds (default) ignore this block.
if(FAISS_WITH_DOCA AND DOCA_FOUND)
  set(DOCA_C_SRCS
    ${DOCA_HOME}/applications/common/comch_utils.c
    ${DOCA_HOME}/applications/common/utils.c
    ${DOCA_HOME}/applications/common/pack.c
    ${DOCA_HOME}/samples/common.c
    ${DOCA_HOME}/samples/doca_common/pe_common.c
    ${DOCA_HOME}/samples/doca_common/common_common.c
    ${DOCA_HOME}/samples/doca_dma/dma_common.c
    ${CMAKE_CURRENT_SOURCE_DIR}/dma_copy_core_modified.c
  )
  list(APPEND PLUGIN_SRCS ${DOCA_C_SRCS})
  set_source_files_properties(${DOCA_C_SRCS} PROPERTIES
    LANGUAGE C
    COMPILE_FLAGS "-Wno-deprecated-declarations -std=c99 -D_POSIX_C_SOURCE=199309L"
  )
  message(STATUS "VeX Plugin: DOCA (DPU) sources enabled")
endif()

target_sources(faiss PRIVATE ${PLUGIN_SRCS})
target_include_directories(faiss PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

if(FAISS_WITH_DOCA AND DOCA_FOUND)
  target_include_directories(faiss PRIVATE ${DOCA_INCLUDE_DIRS})
  target_link_libraries(faiss PRIVATE ${DOCA_LIBRARIES})
endif()

foreach(header ${PLUGIN_HEADERS})
  install(FILES ${header} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/faiss/plugin)
endforeach()

message(STATUS "VeX Plugin: custom indices added (IndexShardBase, IndexKmeansShard, "
               "IndexKmeansShardFinegrained, IndexRandomlyShard)")
