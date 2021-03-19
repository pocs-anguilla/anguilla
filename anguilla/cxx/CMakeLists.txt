set(ANGUILLA_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(MSVC)
  #set(ANGUILLA_COMPILE_OPTIONS /WX)
else()
  set(ANGUILLA_COMPILE_OPTIONS -Wconversion -Wpedantic -Wall)
  #set(ANGUILLA_COMPILE_OPTIONS -Werror ${ANGUILLA_COMPILE_OPTIONS})
endif()

#########################
### Hypervolume extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_hypervolume" MODULE NO_EXTRAS "src/_hypervolume.cpp")
  set_target_properties("_hypervolume" PROPERTIES DEBUG_POSTFIX "")
  target_link_libraries("_hypervolume" PRIVATE ${Asan_LIBRARY})
else()
  pybind11_add_module("_hypervolume" MODULE "src/_hypervolume.cpp")
endif()

target_compile_options("_hypervolume" PRIVATE ${ANGUILLA_COMPILE_OPTIONS})
target_compile_definitions("_hypervolume" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_hypervolume" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${BTREE_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_hypervolume" PRIVATE ${xtensor_INCLUDE_DIRS})
if(WITH_XSIMD)
  target_include_directories("_hypervolume" PRIVATE ${xsimd_INCLUDE_DIRS})
  target_link_libraries("_hypervolume" PUBLIC xsimd)
endif()
target_link_libraries("_hypervolume" PUBLIC xtensor)
target_link_libraries("_hypervolume" PUBLIC pyxtensor)
install(TARGETS "_hypervolume" LIBRARY DESTINATION "hypervolume")

#######################
### Dominance extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_dominance" MODULE NO_EXTRAS "src/_dominance.cpp")
  set_target_properties("_dominance" PROPERTIES DEBUG_POSTFIX "")
  target_link_libraries("_dominance" PRIVATE ${Asan_LIBRARY})
else()
  pybind11_add_module("_dominance" MODULE "src/_dominance.cpp")
endif()

target_compile_options("_dominance" PRIVATE ${ANGUILLA_COMPILE_OPTIONS})
target_compile_definitions("_dominance" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_dominance" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${BOOST_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_dominance" PRIVATE ${xtensor_INCLUDE_DIRS})
if(WITH_XSIMD)
  target_include_directories("_dominance" PRIVATE ${xsimd_INCLUDE_DIRS})
  target_link_libraries("_dominance" PUBLIC xsimd)
endif()
target_link_libraries("_dominance" PUBLIC xtensor)
target_link_libraries("_dominance" PUBLIC pyxtensor)
install(TARGETS "_dominance" LIBRARY DESTINATION "dominance")

#####################
### Archive extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_archive" MODULE NO_EXTRAS "src/_archive.cpp")
  set_target_properties("_archive" PROPERTIES DEBUG_POSTFIX "")
  target_link_libraries("_archive" PRIVATE ${Asan_LIBRARY})
else()
  pybind11_add_module("_archive" MODULE "src/_archive.cpp")
endif()

target_compile_options("_archive" PRIVATE ${ANGUILLA_COMPILE_OPTIONS})
target_compile_definitions("_archive" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_archive" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_archive" PRIVATE ${BOOST_INCLUDE_DIRS})
target_include_directories("_archive" PRIVATE ${SPATIAL_INCLUDE_DIRS})
install(TARGETS "_archive" LIBRARY DESTINATION "archive")

########################
### Optimizers extension

if (CMAKE_BUILD_TYPE MATCHES Debug)
  pybind11_add_module("_optimizers" MODULE NO_EXTRAS "src/_optimizers.cpp")
  set_target_properties("_optimizers" PROPERTIES DEBUG_POSTFIX "")
  target_link_libraries("_optimizers" PRIVATE ${Asan_LIBRARY})
else()
  pybind11_add_module("_optimizers" MODULE "src/_optimizers.cpp")
endif()

target_compile_options("_optimizers" PRIVATE ${ANGUILLA_COMPILE_OPTIONS})
target_compile_definitions("_optimizers" PRIVATE VERSION_INFO=${PROJECT_VERSION})
target_include_directories("_optimizers" PRIVATE ${ANGUILLA_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${BTREE_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${xtl_INCLUDE_DIRS})
target_include_directories("_optimizers" PRIVATE ${xtensor_INCLUDE_DIRS})
if(WITH_XSIMD)
  target_include_directories("_optimizers" PRIVATE ${xsimd_INCLUDE_DIRS})
  target_link_libraries("_optimizers" PUBLIC xsimd)
endif()
target_link_libraries("_optimizers" PUBLIC fmt::fmt)
target_link_libraries("_optimizers" PUBLIC xtensor)
target_link_libraries("_optimizers" PUBLIC pyxtensor)
target_link_libraries("_optimizers" PUBLIC xtensor-blas)
if(BLAS_FOUND)
  target_link_libraries("_optimizers" PUBLIC ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
endif()
install(TARGETS "_optimizers" LIBRARY DESTINATION "optimizers")
