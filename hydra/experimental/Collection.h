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
 * Collection.h
 *
 *  Created on: Oct 28, 2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COLLECTION_H_
#define COLLECTION_H_

/**

#include <iostream>
#include <tuple>

#define _DeclareClass(class_name, args...) \
template<typename ...T> \
class_name(std::tuple<T...> const& t) \
{ std::tie(args) = t; } \
template<typename ...T> \
class_name& operator= ( std::tuple<T...> const& t ) \
{std::tie(args) = t;\
return *this; } \
template<typename ...T> \
operator std::tuple<T...> () { return std::make_tuple(args); }



struct Obj
{
    Obj();


    Obj(int a, int b, double c):
    fA(a),
    fB(b),
    fC(c)
    {}

    _DeclareClass(Obj, fA, fB, fC)


  void Print()
  {
      std::cout << " fA = "<< fA << " fB = "<< fB << " fC = "<< fC << std::endl;
  }


  int fA;
  int fB;
  double fC;
};

int main() {

    // ctor
    Obj A(1,1,1);
    A.Print();

    // conversion tuple -> Obj
    A= std::make_tuple(1,2,3);
    A.Print();

    std::tuple<int, int, int> t(0,0,0);
    // conversion Obj -> tuple
    t = A;

    std::cout << " conversion Obj -> tuple " << std::get<0>(t) << "  " <<  std::get<1>(t)  << "  " <<  std::get<2>(t)  << std::endl;




}
 */


namespace hydra {

namespace experimental {


template<typename> struct RemoveBrackets;
template<typename T> struct RemoveBrackets<void (T)> {
    typedef T Type;
};


#define _DeclCollectable(ClassName, ...)\
\
ClassName(typename ClassName::members_type const& tpl)\


}  // namespace experimental

}  // namespace hydra

#endif /* COLLECTION_H_ */
