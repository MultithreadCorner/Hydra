# - Find TCLAP
# Find the TCLAP headers
#
# TCLAP_INCLUDE_DIR - where to find the TCLAP headers
# TCLAP_FOUND       - True if TCLAP is found

if (TCLAP_INCLUDE_DIR)
  # already in cache, be silent
  set (TCLAP_FIND_QUIETLY TRUE)
endif (TCLAP_INCLUDE_DIR)

# find the headers
find_path (TCLAP_INCLUDE_PATH tclap/CmdLine.h
  PATHS
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_INSTALL_PREFIX}/include
  )

# handle the QUIETLY and REQUIRED arguments and set TCLAP_FOUND to
# TRUE if all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (TCLAP
FOUND_VAR TCLAP_FOUND
REQUIRED_VARS TCLAP_INCLUDE_PATH
FAIL_MESSAGE  "TCLAP could not be found.\
Install TCLAP from your distribution's repository or from 'http://tclap.sourceforge.net/'\
 and set TCLAP_INCLUDE_PATH to point to the headers adding \
 '-DTCLAP_INCLUDE_PATH=path_to_tclap_dir' to the cmake command."
)

if (TCLAP_FOUND)
  set (TCLAP_INCLUDE_DIR ${TCLAP_INCLUDE_PATH})
endif (TCLAP_FOUND)

mark_as_advanced(TCLAP_INCLUDE_DIR)
