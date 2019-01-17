#/*----------------------------------------------------------------------------
# *
# *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
# *
# *   This file is part of Hydra Data Analysis Framework.
# *
# *   Hydra is free software: you can redistribute it and/or modify
# *   it under the terms of the GNU General Public License as published by
# *   the Free Software Foundation, either version 3 of the License, or
# *   (at your option) any later version.
# *
# *   Hydra is distributed in the hope that it will be useful,
# *   but WITHOUT ANY WARRANTY; without even the implied warranty of
# *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# *   GNU General Public License for more details.
# *
# *   You should have received a copy of the GNU General Public License
# *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
# *
# *---------------------------------------------------------------------------*/

#
# FindHydra
#
# This module tries to find the Hydra header files and extrats their version.  It
# sets the following variables.
#
# HYDRA_FOUND       -  Set ON if Hydra headers are found, otherwise OFF.
#
# HYDRA_INCLUDE_DIR -  Include directory for hydra header files.  (All header
#                       files will actually be in the hydra subdirectory.)
# HYDRA_VERSION     -  Version of hydra in the form "major.minor.patch".
#

find_path( HYDRA_INCLUDE_DIR
  PATHS
   ${CMAKE_SOURCE_DIR}
    /usr/include
    /usr/local/include
    ${HYDRA_DIR}
  NAMES hydra/Hydra.h
  DOC "Hydra headers"
  )


if( HYDRA_INCLUDE_DIR )
  list( REMOVE_DUPLICATES HYDRA_INCLUDE_DIR )
endif( HYDRA_INCLUDE_DIR )

# Find hydra version
if (HYDRA_INCLUDE_DIR)
  file( STRINGS ${HYDRA_INCLUDE_DIR}/hydra/Hydra.h
    version
    REGEX "#define HYDRA_VERSION[ \t]+([0-9x]+)"
    )
  string( REGEX REPLACE
    "#define HYDRA_VERSION[ \t]+"
    ""
    version
    "${version}"
    )

  string( REGEX MATCH "^[0-9]" major ${version} )
  string( REGEX REPLACE "^${major}00" "" version "${version}" )
  string( REGEX MATCH "^[0-9]" minor ${version} )
  string( REGEX REPLACE "^${minor}0" "" version "${version}" )
  set( HYDRA_VERSION "${major}.${minor}.${version}")
  set( HYDRA_MAJOR_VERSION "${major}")
  set( HYDRA_MINOR_VERSION "${minor}")
endif()

# Check for required components
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args( Hydra
  FOUND_VAR Hydra_FOUND
  REQUIRED_VARS HYDRA_INCLUDE_DIR
  VERSION_VAR HYDRA_VERSION
  )

if(Hydra_FOUND)
  set(HYDRA_INCLUDE_DIRS ${HYDRA_INCLUDE_DIR})
endif()

mark_as_advanced(HYDRA_INCLUDE_DIR)