# TestU01: http://www.iro.umontreal.ca/~simardr/testu01/tu01.html
#
#  TESTU01_FOUND - System has TestU01
#  TESTU01_INCLUDE_DIRS - The TestU01 include directories
#  TESTU01_LIBRARIES - The libraries needed to use TestU01
#
#  Set TESTU01_ROOT before calling find_package to a path to add an additional
#  search path, e.g.,
#
#  Usage:
#
#  set(TESTU01_ROOT "/path/to/custom/testu01") # prefer over system
#  find_package(TestU01)
#  if(TESTU01_FOUND)
#    target_link_libraries (TARGET ${TESTU01_LIBRARIES})
#  endif()

# If already in cache, be silent
if(TESTU01_INCLUDE_DIRS AND TESTU01_LIBRARIES)
  set (TESTU01_FIND_QUIETLY TRUE)
endif()

find_path(TESTU01_INCLUDE_SRES NAMES sres.h
                               HINTS ${TESTU01_ROOT}/include
                                     $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SSTRING NAMES sstring.h
                                  HINTS ${TESTU01_ROOT}/include
                                        $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SKNUTH NAMES sknuth.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SWALK NAMES swalk.h
                                HINTS ${TESTU01_ROOT}/include
                                      $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SMARSA NAMES smarsa.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SCOMP NAMES scomp.h
                                HINTS ${TESTU01_ROOT}/include
                                      $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SSPECTRAL NAMES sspectral.h
                                    HINTS ${TESTU01_ROOT}/include
                                          $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_UNIF01 NAMES unif01.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SNPAIR NAMES snpair.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_GDEF NAMES gdef.h
                               HINTS ${TESTU01_ROOT}/include
                                     $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_GOFW NAMES gofw.h
                               HINTS ${TESTU01_ROOT}/include
                                     $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SVARIA NAMES svaria.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)
find_path(TESTU01_INCLUDE_SWRITE NAMES swrite.h
                                 HINTS ${TESTU01_ROOT}/include
                                       $ENV{TESTU01_ROOT}/include)

set(TESTU01_INCLUDE_DIRS ${TESTU01_INCLUDE_SRES}
                         ${TESTU01_INCLUDE_SSTRING}
                         ${TESTU01_INCLUDE_SKNUTH}
                         ${TESTU01_INCLUDE_SWALK}
                         ${TESTU01_INCLUDE_SMARSA}
                         ${TESTU01_INCLUDE_SCOMP}
                         ${TESTU01_INCLUDE_SSPECTRAL}
                         ${TESTU01_INCLUDE_UNIF01}
                         ${TESTU01_INCLUDE_SNPAIR}
                         ${TESTU01_INCLUDE_GDEF}
                         ${TESTU01_INCLUDE_GOFW}
                         ${TESTU01_INCLUDE_SVARIA}
                         ${TESTU01_INCLUDE_SWRITE})

list(REMOVE_DUPLICATES TESTU01_INCLUDE_DIRS)

if(NOT BUILD_SHARED_LIBS)
  find_library(TESTU01_LIBRARY NAMES libtestu01.so HINTS ${TESTU01_ROOT}/lib64
                                                        $ENV{TESTU01_ROOT}/lib64)
  find_library(TESTU01_PROBDIST_LIBRARY NAMES libprobdist.so
                                        HINTS ${TESTU01_ROOT}/lib64
                                              $ENV{TESTU01_ROOT}/lib64)
  find_library(TESTU01_MYLIB_LIBRARY NAMES libmylib.so
                                     HINTS ${TESTU01_ROOT}/lib64
                                           $ENV{TESTU01_ROOT}/lib64)
else()
  find_library(TESTU01_LIBRARY NAMES testu01 HINTS ${TESTU01_ROOT}/lib64
                                                   $ENV{TESTU01_ROOT}/lib64)
  find_library(TESTU01_PROBDIST_LIBRARY NAMES probdist
                                        HINTS ${TESTU01_ROOT}/lib64
                                              $ENV{TESTU01_ROOT}/lib64)
  find_library(TESTU01_MYLIB_LIBRARY NAMES mylib
                                     HINTS ${TESTU01_ROOT}/lib64
                                           $ENV{TESTU01_ROOT}/lib64)
endif()

set(TESTU01_LIBRARIES ${TESTU01_LIBRARY} ${TESTU01_PROBDIST_LIBRARY}
                      ${TESTU01_MYLIB_LIBRARY})

# Handle the QUIETLY and REQUIRED arguments and set TESTU01_FOUND to TRUE if
# all listed variables are TRUE.
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(TestU01 DEFAULT_MSG TESTU01_LIBRARIES TESTU01_INCLUDE_DIRS)

if(NOT TestU01_FOUND)
  set(TESTU01_INCLUDE_DIRS "")
  set(TESTU01_LIBRARIES "")
endif()

MARK_AS_ADVANCED(TESTU01_INCLUDE_DIRS TESTU01_LIBRARIES)

