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
 * multivector.cpp
 *
 *  Created on: 19/10/2016
 *      Author: Antonio Augusto Alves Junior
 */



#include <thrust/device_malloc_allocator.h>
#include <hydra/experimental/multivector.h>

using namespace std;
using namespace hydra;

int main(int argv, char** argc)
{

	typedef experimental::multivector<thrust::host_vector, std::allocator, double, double> storage;

	storage ms(10);

	cout << ms.size() << endl;

	for(int i =0; i<ms.size(); i++ ){
		ms[i] = thrust::make_tuple(0.5+i, 1.5+i);
	}

	for( auto element:ms )
		std::cout<<element<<std::endl;

}
