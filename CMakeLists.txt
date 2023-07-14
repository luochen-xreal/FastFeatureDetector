cmake_minimum_required(VERSION 3.1)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(main CXX)

add_executable(main main.cpp)
add_executable(ffdtest ffdtest.cc)

include_directories(
${PROJECT_SOURCE_DIR}
${OpenCV_INCLUDE_DIRS}
)

# find_package(Eigen3 REQUIRED NO_MODULE)
# message("EIGEN3_INCLUDE_DIR:" ${EIGEN3_INCLUDE_DIR})

# find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
# if(NOT OpenCV_FOUND)
#    find_package(OpenCV 4)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV not found.")
#    endif()
# endif()


add_library(FFD SHARED
ffd.cc

)
target_link_libraries(FFD ${OpenCV_LIBS})

target_include_directories(main PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main ${OpenCV_LIBS})


target_include_directories(ffdtest PUBLIC ${OpenCV_INCLUDE_DIRS})

target_link_libraries(ffdtest ${OpenCV_LIBS} FFD)