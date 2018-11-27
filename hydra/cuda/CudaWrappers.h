/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * APIWrappers.h
 *
 *  Created on: Nov 27, 2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef APIWRAPPERS_H_
#define APIWRAPPERS_H_


#include <hydra/detail/Config.h>


/**
 * Set of wrapper functions mimicking std memory management interface for CUDA devices.
 * Pointers are assumed to be always defined on memory address space.
 * Call from host side.
 *
 */

namespace hydra {

namespace cuda {

void* malloc( size_t size )
{
	    void *ptr;
	    if (cudaMalloc(&ptr, len) == cudaSuccess) return ptr;
	    return 0;
}

void free( void* ptr ){
	 cudaFree( devPtr);
}


void* memset( void* dest, int ch, size_t count  )
{
	    if (cudaMemset(dest, ch, count) == cudaSuccess) return dest;
	    return 0;
}

void* memcpy( void* dest, const void* src, size_t count ){
	 if( cudaMemcpy( dst, src, count, cudaMemcpyDeviceToDevice) == cudaSuccess) return dest;
	 return 0;
}


}  // namespace cuda

}  // namespace hydra



#endif /* APIWRAPPERS_H_ */
