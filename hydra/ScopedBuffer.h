/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * ScopedBuffer.h
 *
 *  Created on: 26/12/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SCOPEDBUFFER_H_
#define SCOPEDBUFFER_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>

#include <hydra/detail/external/thrust/memory.h>
#include <memory>
#include <type_traits>


namespace hydra {

template<typename T, typename BACKEND>
class ScopedBuffer;

template<typename T,detail::Backend BACKEND>
class ScopedBuffer<T, detail::BackendPolicy<BACKEND> >
{
	typedef decltype(std::declval<detail::BackendPolicy<BACKEND>>().backend) system_type;
	typedef HYDRA_EXTERNAL_NS::thrust::pointer<T, typename std::remove_const<system_type>::type >  pointer_type;

public:

	ScopedBuffer()=delete;

	ScopedBuffer(size_t n):
		fSize(n)
	{
		auto policy = detail::BackendPolicy<BACKEND>{};
		auto buffer = HYDRA_EXTERNAL_NS::thrust::get_temporary_buffer<T>(policy, size);
		fPointer    = buffer.first;

	}

	ScopedBuffer(ScopedBuffer<T, detail::BackendPolicy<BACKEND>> const& other):
		fSize( other.GetSize()),
		fPointer( other.GetPointer())
	{}

	ScopedBuffer<T, detail::BackendPolicy<BACKEND>>&
	operator=(ScopedBuffer<T, detail::BackendPolicy<BACKEND>> const& other){

		if(this!=&other) return *this;

		fSize    = other.GetSize();
		fPointer = other.GetPointer();

		return *this;
	}

	const pointer_type& GetPointer() const
	{
		return fPointer;
	}

	size_t GetSize() const
	{
		return fSize;
	}

	~ScopedBuffer(){

		using HYDRA_EXTERNAL_NS::thrust::return_temporary_buffer;

		return_temporary_buffer(system_type(), fPointer);
	}

private:

	size_t    fSize;
	pointer_type fPointer;
};


}  // namespace hydra

#endif /* SCOPEDBUFFER_H_ */
