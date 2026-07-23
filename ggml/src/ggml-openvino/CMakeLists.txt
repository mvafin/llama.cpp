find_package(OpenVINO REQUIRED)
find_package(OpenCL REQUIRED)

include("${OpenVINO_DIR}/../3rdparty/tbb/lib/cmake/TBB/TBBConfig.cmake")

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

# The GGUF frontend ships as a linkable OpenVINO frontend (openvino::frontend::gguf), installed
# alongside the runtime and exported via find_package(OpenVINO). It is hidden from
# FrontEndManager, so it is used only through this direct linkage. It is currently shipped only
# in the OpenVINO nightly packages (not the numbered releases), so fail early with a clear
# message if the found OpenVINO does not export it rather than emitting a cryptic link error.
if (NOT TARGET openvino::frontend::gguf)
    message(FATAL_ERROR
        "The OpenVINO found at '${OpenVINO_DIR}' does not provide the GGUF frontend "
        "(openvino::frontend::gguf), which the OpenVINO backend links against. Use an OpenVINO "
        "nightly package that ships it (see docs/backend/OPENVINO.md); the numbered releases do "
        "not, and it is only just landing on OpenVINO master. Until then, build OpenVINO from a "
        "source branch with -DENABLE_OV_GGUF_FRONTEND=ON.")
endif()

target_link_libraries(ggml-openvino PRIVATE
    openvino::runtime openvino::frontend::gguf TBB::tbb OpenCL::OpenCL)

if (GGML_OPENVINO)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    else()
        message(FATAL_ERROR "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()
