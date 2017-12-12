cmake_minimum_required(VERSION 2.8.12)
project(saxpy)

SET(SOURCE_DIR "src")
include_directories(${SOURCE_DIR})
SET(SOURCES "${SOURCE_DIR}/znorm.cpp")

add_subdirectory(pybind11)
pybind11_add_module(_saxpy ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
