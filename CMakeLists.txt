cmake_minimum_required(VERSION 3.8)
project(navbot_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)


install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

ament_python_install_package(${PROJECT_NAME})
install(PROGRAMS
  scripts/test.py
  scripts/robot_ros_node.py
  scripts/controller_test.py
  scripts/straight.py
  scripts/publish_guides.py
  scripts/logger.py
  scripts/plot_odometry.py
  scripts/path_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
