set(ANGUILLA_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

#########################
### Hypervolume extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_hypervolume" MODULE NO_EXTRAS "src/_hypervolume.cpp")
  set_target_properties("_hypervolume" PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries("_hypervolume" PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module("_hypervolume" MODULE "src/_hypervolume.cpp")
endif()

#if(NOT MSVC)
#  target_compile_options("_hypervolume" BEFORE PRIVATE "-Wconversion" "-Werror")
#else()
  # TODO: enable and check warnings
  #target_compile_options("_hypervolume" BEFORE PRIVATE "/WX")
#endif()

target_compile_definitions("_hypervolume" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_hypervolume" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${BTREE_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${xtensor_INCLUDE_DIRS})
target_link_libraries("_hypervolume" PUBLIC xtensor pyxtensor)

install(TARGETS "_hypervolume" LIBRARY DESTINATION ".")

#######################
### Dominance extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_dominance" MODULE NO_EXTRAS "src/_dominance.cpp")
  set_target_properties("_dominance" PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries("_dominance" PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module("_dominance" MODULE "src/_dominance.cpp")
endif()

if(NOT MSVC)
  target_compile_options("_dominance" BEFORE PRIVATE "-Werror")
else()
  # TODO: enable and check warnings
  #target_compile_options("_dominance" BEFORE PRIVATE "/WX")
endif()

target_compile_definitions("_dominance" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_dominance" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${BOOST_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${xtensor_INCLUDE_DIRS})
target_link_libraries("_dominance" PUBLIC xtensor pyxtensor)


install(TARGETS "_dominance" LIBRARY DESTINATION ".")

#####################
### Archive extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_archive" MODULE NO_EXTRAS "src/_archive.cpp")
  set_target_properties("_archive" PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries("_archive" PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module("_archive" MODULE "src/_archive.cpp")
endif()

target_compile_definitions("_archive" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_archive" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_archive" PRIVATE ${BOOST_INCLUDE_DIRS})
target_include_directories("_archive" PRIVATE ${SPATIAL_INCLUDE_DIRS})

if(NOT MSVC)
  target_compile_options("_archive" BEFORE PRIVATE "-Werror")
else()
  # TODO: enable and check warnings
  #target_compile_options("_archive" BEFORE PRIVATE "/WX")
endif()

install(TARGETS "_archive" LIBRARY DESTINATION ".")

###############################################
### Bindings for Shark's Hypervolume (optional)

if(WITH_SHARK_BINDINGS)
    find_package(Shark REQUIRED)
    include(${SHARK_USE_FILE})
    find_package(Boost REQUIRED COMPONENTS regex system filesystem serialization)

    pybind11_add_module("_shark_hypervolume" MODULE "src/_shark_hypervolume.cpp")
    target_include_directories("_shark_hypervolume" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
    target_link_libraries("_shark_hypervolume" PRIVATE ${SHARK_LIBRARIES})
    target_link_libraries("_shark_hypervolume" PRIVATE ${Boost_LIBRARIES})

    install(TARGETS "_shark_hypervolume" LIBRARY DESTINATION ".")
endif()

########################
### Optimizers extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_optimizers" MODULE NO_EXTRAS "src/_optimizers.cpp")
  set_target_properties("_optimizers" PROPERTIES DEBUG_POSTFIX "")
  if(NOT MSVC)
    find_library(Asan_LIBRARY asan REQUIRED)
    target_link_libraries("_optimizers" PRIVATE ${Asan_LIBRARY})
  endif()
else()
  pybind11_add_module("_optimizers" MODULE "src/_optimizers.cpp")
endif()

#if(NOT MSVC)
#  target_compile_options("_optimizers" BEFORE PRIVATE "-Wconversion" "-Werror")
#else()
  # TODO: enable and check warnings
  #target_compile_options("_optimizers" BEFORE PRIVATE "/WX")
#endif()

target_compile_definitions("_optimizers" PRIVATE VERSION_INFO=${PROJECT_VERSION} HAVE_CBLAS=1)

target_include_directories("_optimizers" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${xtensor_INCLUDE_DIRS})

target_link_libraries("_optimizers" PUBLIC xtensor xtensor-blas pyxtensor)
if(BLAS_FOUND)
  target_link_libraries("_optimizers" PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
#target_link_libraries("_optimizers" PUBLIC fmt::fmt-header-only)
target_link_libraries("_optimizers" PUBLIC fmt::fmt)

install(TARGETS "_optimizers" LIBRARY DESTINATION ".")