/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 Antonio Augusto Alves Junior
 *
 *   This file is part of Hydra Data Analysis Framework.
 *
 *   Hydra is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   Hydra is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with Hydra.  If not, see <http://www.gnu.org/licenses/>.
 *
 *---------------------------------------------------------------------------*/

/*
 * Print.h
 *
 *  Created on: 27/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PRINT_H_
#define PRINT_H_

#include <hydra/detail/utility/StreamSTL.h>
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <iostream>



namespace hydra {


enum {
	INFO=0, WARNING=1, ERROR=2
};

#ifdef DEBUG
int PrintLevel = INFO;
#else
int PrintLevel = WARNING;
#endif

class Print {


public:
   // set print level and return the previous one
   static int SetLevel(int level){
	   int prevLevel  = PrintLevel;
	   PrintLevel = level;
	   return prevLevel;
	}

   // return current level
   static int Level() {
	   return PrintLevel;
	}

   static const char* Label(int level){

	   const char *labels[3] = {"Info", "Warning", "Error"};
	   return labels[level];

   }

};


}  // namespace hydra

#define HYDRA_LOG(level, str) \
   if ( level >= Print::Level()) HYDRA_OS << "\033[1;34m" << "\nHydra["<< Print::Label(level) << "] from: \n"\
   << "\033[1;32m" << __PRETTY_FUNCTION__ << '\n' << "\033[1;34m" << "FILE: "<< "\033[1;32m"  << __FILE__ << "\n"\
   << "\033[1;34m" << "LINE :"<< "\033[1;32m"<<__LINE__ << "\033[1;32m"<< "\n" << "\033[1;34m" \
   << "MESSAGE: " << "\033[1;31m" << str <<"\033[0m"<< std::endl << std::endl;

#define HYDRA_CALLER \
  std::cout << "\033[1;32m"<< "|--- Hydra ---> "\
  << __PRETTY_FUNCTION__<<"\033[0m"<< std::endl

#define HYDRA_MSG \
  std::cout << "\033[1;33m"<< "|--- Hydra --->: " <<"\033[0;36m"

#define HYDRA_SPACED_MSG \
  std::cout << "\033[1;33m"<< "|--- Hydra ------>: " <<"\033[0;36m"

#define HYDRA_ENDL "\033[0m"<< std::endl


#endif /* PRINT_H_ */
