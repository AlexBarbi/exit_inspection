cmake_minimum_required(VERSION 3.8)
project(exit_inspection)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})
install(DIRECTORY ${PROJECT_NAME}/assets DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})

install(PROGRAMS
  scripts/exit_landmark_mapper.py
  scripts/finalize_exits.py
  scripts/check_exits_gt.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config worlds models DESTINATION share/${PROJECT_NAME})

ament_package()
