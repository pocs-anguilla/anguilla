set(MODULE_NAME "archive")
set(INTERNAL_MODULE_NAME "_${MODULE_NAME}")

set(MODULE_SRC
    include/anguilla/archive/archive.hpp
    include/anguilla/archive/archive_kd.hpp
    include/anguilla/archive/individual.hpp
    include/anguilla/archive/parameters.hpp
    "src/${INTERNAL_MODULE_NAME}.cpp"
)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module(${INTERNAL_MODULE_NAME} MODULE NO_EXTRAS ${MODULE_SRC})
  set_target_properties(${INTERNAL_MODULE_NAME} PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries(${INTERNAL_MODULE_NAME} PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module(${INTERNAL_MODULE_NAME} MODULE ${MODULE_SRC})
endif()

target_compile_definitions(${INTERNAL_MODULE_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(${INTERNAL_MODULE_NAME} PRIVATE include)
target_include_directories(${INTERNAL_MODULE_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
target_include_directories(${INTERNAL_MODULE_NAME} PRIVATE ${BOOST_INCLUDE_DIRS})
target_include_directories(${INTERNAL_MODULE_NAME} PRIVATE ${SPATIAL_INCLUDE_DIRS})

if(NOT MSVC)
  target_compile_options(${INTERNAL_MODULE_NAME} BEFORE PRIVATE "-Werror")
else()
  # TODO: enable and check warnings
  #target_compile_options(${INTERNAL_MODULE_NAME} BEFORE PRIVATE "/WX")
endif()

install(TARGETS ${INTERNAL_MODULE_NAME} LIBRARY DESTINATION ${MODULE_NAME})
