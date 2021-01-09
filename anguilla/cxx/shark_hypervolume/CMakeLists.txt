find_package(Shark REQUIRED)
include(${SHARK_USE_FILE})
find_package(Boost REQUIRED COMPONENTS regex system filesystem serialization)

set(SHARK_HV_MODULE_NAME "_shark_hypervolume")

set(HV_MODULE_SRC
    include/pyshark.hpp
    "src/${SHARK_HV_MODULE_NAME}.cpp"
)

pybind11_add_module(${SHARK_HV_MODULE_NAME} MODULE ${HV_MODULE_SRC})
target_include_directories(${SHARK_HV_MODULE_NAME} PRIVATE include)
target_link_libraries(${SHARK_HV_MODULE_NAME} PRIVATE ${SHARK_LIBRARIES})
target_link_libraries(${SHARK_HV_MODULE_NAME} PRIVATE ${Boost_LIBRARIES})

install(TARGETS ${SHARK_HV_MODULE_NAME} LIBRARY DESTINATION hypervolume)
