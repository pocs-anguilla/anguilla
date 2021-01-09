set(HV_MODULE_NAME "hypervolume")
set(HV_INTERNAL_MODULE_NAME "_${HV_MODULE_NAME}")

set(HV_MODULE_SRC
    include/common.hpp
    include/hv2d.hpp
    include/hv3d.hpp
    include/hvc2d.hpp
    include/hvc3d.hpp
    "src/${HV_INTERNAL_MODULE_NAME}.cpp"
)

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module(${HV_INTERNAL_MODULE_NAME} MODULE NO_EXTRAS ${HV_MODULE_SRC})
  set_target_properties(${HV_INTERNAL_MODULE_NAME} PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries(${HV_INTERNAL_MODULE_NAME} PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module(${HV_INTERNAL_MODULE_NAME} MODULE ${HV_MODULE_SRC})
endif()

target_compile_definitions(${HV_INTERNAL_MODULE_NAME} PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories(${HV_INTERNAL_MODULE_NAME} PRIVATE include)
target_include_directories(${HV_INTERNAL_MODULE_NAME} PRIVATE ${BTREE_INCLUDE_DIRS})

if(NOT MSVC)
  target_compile_options(${HV_INTERNAL_MODULE_NAME} BEFORE PRIVATE "-Werror")
endif()

install(TARGETS ${HV_INTERNAL_MODULE_NAME} LIBRARY DESTINATION ${HV_MODULE_NAME})
