# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build

# Include any dependencies generated for this target.
include CMakeFiles/pase.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/pase.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/pase.dir/flags.make

CMakeFiles/pase.dir/source/pase_config.c.o: CMakeFiles/pase.dir/flags.make
CMakeFiles/pase.dir/source/pase_config.c.o: ../source/pase_config.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/pase.dir/source/pase_config.c.o"
	mpicc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/pase.dir/source/pase_config.c.o   -c /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_config.c

CMakeFiles/pase.dir/source/pase_config.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pase.dir/source/pase_config.c.i"
	mpicc  $(C_DEFINES) $(C_FLAGS) -E /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_config.c > CMakeFiles/pase.dir/source/pase_config.c.i

CMakeFiles/pase.dir/source/pase_config.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pase.dir/source/pase_config.c.s"
	mpicc  $(C_DEFINES) $(C_FLAGS) -S /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_config.c -o CMakeFiles/pase.dir/source/pase_config.c.s

CMakeFiles/pase.dir/source/pase_config.c.o.requires:
.PHONY : CMakeFiles/pase.dir/source/pase_config.c.o.requires

CMakeFiles/pase.dir/source/pase_config.c.o.provides: CMakeFiles/pase.dir/source/pase_config.c.o.requires
	$(MAKE) -f CMakeFiles/pase.dir/build.make CMakeFiles/pase.dir/source/pase_config.c.o.provides.build
.PHONY : CMakeFiles/pase.dir/source/pase_config.c.o.provides

CMakeFiles/pase.dir/source/pase_config.c.o.provides.build: CMakeFiles/pase.dir/source/pase_config.c.o

CMakeFiles/pase.dir/source/pase_matrix.c.o: CMakeFiles/pase.dir/flags.make
CMakeFiles/pase.dir/source/pase_matrix.c.o: ../source/pase_matrix.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/pase.dir/source/pase_matrix.c.o"
	mpicc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/pase.dir/source/pase_matrix.c.o   -c /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix.c

CMakeFiles/pase.dir/source/pase_matrix.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pase.dir/source/pase_matrix.c.i"
	mpicc  $(C_DEFINES) $(C_FLAGS) -E /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix.c > CMakeFiles/pase.dir/source/pase_matrix.c.i

CMakeFiles/pase.dir/source/pase_matrix.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pase.dir/source/pase_matrix.c.s"
	mpicc  $(C_DEFINES) $(C_FLAGS) -S /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix.c -o CMakeFiles/pase.dir/source/pase_matrix.c.s

CMakeFiles/pase.dir/source/pase_matrix.c.o.requires:
.PHONY : CMakeFiles/pase.dir/source/pase_matrix.c.o.requires

CMakeFiles/pase.dir/source/pase_matrix.c.o.provides: CMakeFiles/pase.dir/source/pase_matrix.c.o.requires
	$(MAKE) -f CMakeFiles/pase.dir/build.make CMakeFiles/pase.dir/source/pase_matrix.c.o.provides.build
.PHONY : CMakeFiles/pase.dir/source/pase_matrix.c.o.provides

CMakeFiles/pase.dir/source/pase_matrix.c.o.provides.build: CMakeFiles/pase.dir/source/pase_matrix.c.o

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o: CMakeFiles/pase.dir/flags.make
CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o: ../source/pase_matrix_hypre.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o"
	mpicc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o   -c /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix_hypre.c

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pase.dir/source/pase_matrix_hypre.c.i"
	mpicc  $(C_DEFINES) $(C_FLAGS) -E /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix_hypre.c > CMakeFiles/pase.dir/source/pase_matrix_hypre.c.i

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pase.dir/source/pase_matrix_hypre.c.s"
	mpicc  $(C_DEFINES) $(C_FLAGS) -S /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_matrix_hypre.c -o CMakeFiles/pase.dir/source/pase_matrix_hypre.c.s

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.requires:
.PHONY : CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.requires

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.provides: CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.requires
	$(MAKE) -f CMakeFiles/pase.dir/build.make CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.provides.build
.PHONY : CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.provides

CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.provides.build: CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o

CMakeFiles/pase.dir/source/pase_vector.c.o: CMakeFiles/pase.dir/flags.make
CMakeFiles/pase.dir/source/pase_vector.c.o: ../source/pase_vector.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles $(CMAKE_PROGRESS_4)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/pase.dir/source/pase_vector.c.o"
	mpicc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/pase.dir/source/pase_vector.c.o   -c /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector.c

CMakeFiles/pase.dir/source/pase_vector.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pase.dir/source/pase_vector.c.i"
	mpicc  $(C_DEFINES) $(C_FLAGS) -E /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector.c > CMakeFiles/pase.dir/source/pase_vector.c.i

CMakeFiles/pase.dir/source/pase_vector.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pase.dir/source/pase_vector.c.s"
	mpicc  $(C_DEFINES) $(C_FLAGS) -S /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector.c -o CMakeFiles/pase.dir/source/pase_vector.c.s

CMakeFiles/pase.dir/source/pase_vector.c.o.requires:
.PHONY : CMakeFiles/pase.dir/source/pase_vector.c.o.requires

CMakeFiles/pase.dir/source/pase_vector.c.o.provides: CMakeFiles/pase.dir/source/pase_vector.c.o.requires
	$(MAKE) -f CMakeFiles/pase.dir/build.make CMakeFiles/pase.dir/source/pase_vector.c.o.provides.build
.PHONY : CMakeFiles/pase.dir/source/pase_vector.c.o.provides

CMakeFiles/pase.dir/source/pase_vector.c.o.provides.build: CMakeFiles/pase.dir/source/pase_vector.c.o

CMakeFiles/pase.dir/source/pase_vector_hypre.c.o: CMakeFiles/pase.dir/flags.make
CMakeFiles/pase.dir/source/pase_vector_hypre.c.o: ../source/pase_vector_hypre.c
	$(CMAKE_COMMAND) -E cmake_progress_report /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles $(CMAKE_PROGRESS_5)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building C object CMakeFiles/pase.dir/source/pase_vector_hypre.c.o"
	mpicc  $(C_DEFINES) $(C_FLAGS) -o CMakeFiles/pase.dir/source/pase_vector_hypre.c.o   -c /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector_hypre.c

CMakeFiles/pase.dir/source/pase_vector_hypre.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/pase.dir/source/pase_vector_hypre.c.i"
	mpicc  $(C_DEFINES) $(C_FLAGS) -E /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector_hypre.c > CMakeFiles/pase.dir/source/pase_vector_hypre.c.i

CMakeFiles/pase.dir/source/pase_vector_hypre.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/pase.dir/source/pase_vector_hypre.c.s"
	mpicc  $(C_DEFINES) $(C_FLAGS) -S /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/source/pase_vector_hypre.c -o CMakeFiles/pase.dir/source/pase_vector_hypre.c.s

CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.requires:
.PHONY : CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.requires

CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.provides: CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.requires
	$(MAKE) -f CMakeFiles/pase.dir/build.make CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.provides.build
.PHONY : CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.provides

CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.provides.build: CMakeFiles/pase.dir/source/pase_vector_hypre.c.o

# Object files for target pase
pase_OBJECTS = \
"CMakeFiles/pase.dir/source/pase_config.c.o" \
"CMakeFiles/pase.dir/source/pase_matrix.c.o" \
"CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o" \
"CMakeFiles/pase.dir/source/pase_vector.c.o" \
"CMakeFiles/pase.dir/source/pase_vector_hypre.c.o"

# External object files for target pase
pase_EXTERNAL_OBJECTS =

libpase.a: CMakeFiles/pase.dir/source/pase_config.c.o
libpase.a: CMakeFiles/pase.dir/source/pase_matrix.c.o
libpase.a: CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o
libpase.a: CMakeFiles/pase.dir/source/pase_vector.c.o
libpase.a: CMakeFiles/pase.dir/source/pase_vector_hypre.c.o
libpase.a: CMakeFiles/pase.dir/build.make
libpase.a: CMakeFiles/pase.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking C static library libpase.a"
	$(CMAKE_COMMAND) -P CMakeFiles/pase.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/pase.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/pase.dir/build: libpase.a
.PHONY : CMakeFiles/pase.dir/build

CMakeFiles/pase.dir/requires: CMakeFiles/pase.dir/source/pase_config.c.o.requires
CMakeFiles/pase.dir/requires: CMakeFiles/pase.dir/source/pase_matrix.c.o.requires
CMakeFiles/pase.dir/requires: CMakeFiles/pase.dir/source/pase_matrix_hypre.c.o.requires
CMakeFiles/pase.dir/requires: CMakeFiles/pase.dir/source/pase_vector.c.o.requires
CMakeFiles/pase.dir/requires: CMakeFiles/pase.dir/source/pase_vector_hypre.c.o.requires
.PHONY : CMakeFiles/pase.dir/requires

CMakeFiles/pase.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/pase.dir/cmake_clean.cmake
.PHONY : CMakeFiles/pase.dir/clean

CMakeFiles/pase.dir/depend:
	cd /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130 /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130 /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build /home/hongqichen/software/hypre-2.11.2/src/examples/pase1130/build/CMakeFiles/pase.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/pase.dir/depend
