# Daniel Peek qer419
# Michael Canas ohh135
# CS3793 Assignment 05
# 11/8/2018
# CMake file for project

cmake_minimum_required(VERSION 3.12)
project(CS3793_assignment_05 C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(CS3793_assignment_05
        bp.c
        bp.h
        prt.c
        readMNIST.c)

target_link_libraries(CS3793_assignment_05 m)
