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
 * Function.h
 *
 *  Created on: 05/05/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef FUNCTION_H_
#define FUNCTION_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/Parameters.h>
//#include <hydra/UserParameters.h>

#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <array>
#include <initializer_list>
#include <memory>


namespace hydra
{

/**
 * @ingroup functor
 * @brief Base class for all functors in hydra.
 * @tparam Functor is "bare" c++ functor implementing the template<typename T> Evaluate(T x) method.
 * @tparam ReturnType type returned by the functor' operator(). Same type returned by the "bare" c++ functor Evaluate() method.
 * @tparam NPARAM number of parameters of the functor.
 */
template<typename Functor, typename ReturnType, size_t NPARAM>
class  BaseFunctor : public detail::Parameters<NPARAM>
{

public:

	//tag
    typedef   void hydra_functor_tag;
	typedef   ReturnType return_type;
	typedef   std::true_type is_functor;

	/**
	 * Default constructor
	 */
	__hydra_host__  __hydra_device__
	explicit BaseFunctor():
		detail::Parameters<NPARAM>(),
		fCacheIndex(-1),
		fCached(0),
		fNorm(1.0),
		_par(*this)
	{}


    /**
     * @brief Constructor taking a list of parameters
     * @param init_parameters std::initializer_list<Parameter> with the parameters of the functor.
     */
	BaseFunctor(std::initializer_list<Parameter> init_parameters):
	detail::Parameters<NPARAM>( init_parameters ),
	fCacheIndex(-1),
	fCached(0),
	fNorm(1.0),
	_par(*this)
	{}

	/**
	 * @brief Constructor taking std::array of parameters
	 * @param init_parameters std::array<Parameter,NPARAM> with the parameters of the functor.
	 */
	BaseFunctor(std::array<Parameter,NPARAM> const& init_parameters):
		detail::Parameters<NPARAM>( init_parameters ),
		fCacheIndex(-1),
		fCached(0),
		fNorm(1.0),
		_par(*this)
		{ }


	/**
	 * @brief Copy constructor
	 */
	__hydra_host__ __hydra_device__
	BaseFunctor(BaseFunctor<Functor,ReturnType, NPARAM> const& other):
	detail::Parameters<NPARAM>( other),
	fCacheIndex( other.GetCacheIndex() ),
	fCached( other.IsCached() ),
	fNorm(other.GetNorm()),
	_par(*this)
	{ }

	/**
	 * @brief Assignment operator
	 */
	__hydra_host__ __hydra_device__
	inline BaseFunctor<Functor,ReturnType, NPARAM>&
	operator=(BaseFunctor<Functor, ReturnType, NPARAM> const & other )
	{
		if(this != &other)
		{
			detail::Parameters<NPARAM>::operator=( other );
			this->fCacheIndex     = other.GetCacheIndex();
			this->fCached         = other.IsCached();
			this->fNorm = other.GetNorm();

			_par=*this;

		}
		return *this;
	}


	__hydra_host__ __hydra_device__
	inline Functor& GetFunctor() {return *static_cast<Functor*>(this);}

	__hydra_host__ __hydra_device__
	inline int GetCacheIndex() const { return this->fCacheIndex; }

	__hydra_host__ __hydra_device__
	inline void SetCacheIndex(int index) {fCacheIndex = index;}

	__hydra_host__ __hydra_device__
	inline bool IsCached() const
	{ return this->fCached;}

	__hydra_host__ __hydra_device__
	inline void SetCached(bool cached=true)
	{ fCached = cached; }


	/**
	 * @brief Print registered parameters.
	 */
	void PrintRegisteredParameters()
	{

		HYDRA_CALLER ;
		HYDRA_MSG <<HYDRA_ENDL;
		/*
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		for(size_t i=0; i<parameter_count; i++ )

		HYDRA_MSG <<"  >> Parameter " << i <<") "<< fParameters[i] << HYDRA_ENDL;
		*/
		this->PrintParameters();

		HYDRA_MSG <<"Normalization " << fNorm << HYDRA_ENDL;
		HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}


	__hydra_host__ __hydra_device__
	inline GReal_t GetNorm() const {
		return fNorm;
	}

	__hydra_host__ __hydra_device__
	inline void SetNorm(GReal_t norm) {
		fNorm = norm;
	}

	template<typename T>
	__hydra_host__  __hydra_device__
	inline return_type operator()(unsigned int n, T* x)  const
	{
		return static_cast<const Functor*>(this)->Evaluate(n,x);
	}

	template<typename T>
	__hydra_host__ __hydra_device__
	inline return_type operator()( T&&  x )  const
	{
		return  interface( std::forward<T>(x));
	}

	template<typename T1, typename T2>
	__hydra_host__ __hydra_device__
	inline return_type operator()( T1&& x, T2&& cache)  const
	{
		return fCached ? detail::extract<return_type, T2 >(fCacheIndex, std::forward<T2>(cache)):
						operator()<T1>( std::forward<T1>(x) );
	}

private:


	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	! ( detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::tuple,
			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
				typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference< T>::type
			>::type >::value ||
	    detail::is_instantiation_of< HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references,
	        typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	        	typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
	        >::type >::value ) , return_type>::type
	interface(T&& x)  const
	{
		//fNArgs=1;
		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type >::type _x;

		_x=x;
		return static_cast<const Functor*>(this)->Evaluate(1, &_x);
	}


	template<typename T>
	__hydra_host__ __hydra_device__
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(
			  detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::tuple,
			  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			  	  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
			  >::type >::value ||
			  detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::detail::tuple_of_iterator_references,
			  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
			  	  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
			   >::type >::value ) &&
	        detail::is_homogeneous<
	        	typename HYDRA_EXTERNAL_NS::thrust::tuple_element<0,
	        		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	        			typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
	        		>::type
	        	>::type,
	        	typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	        		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
	        	>::type
	        >::value, return_type>::type
	interface(T&& x)  const
	{
		typedef  typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type>::type Tprime;
		typedef typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<typename HYDRA_EXTERNAL_NS::thrust::tuple_element<0, Tprime>::type>::type first_type;
		constexpr size_t N = HYDRA_EXTERNAL_NS::thrust::tuple_size< Tprime >::value;

		first_type Array[ N ];

		detail::tupleToArray(x, &Array[0] );
		//fNArgs=N;
		return static_cast<const Functor*>(this)->Evaluate(N, &Array[0]);


	}

	template<typename T >
	__hydra_host__ __hydra_device__
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<
	detail::is_instantiation_of<HYDRA_EXTERNAL_NS::thrust::tuple,
		typename std::remove_reference<T>::type >::value &&
	!(detail::is_homogeneous<
	    typename HYDRA_EXTERNAL_NS::thrust::tuple_element< 0,
	    	typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	    		typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
	    	>::type
	    >::type,
	    typename HYDRA_EXTERNAL_NS::thrust::detail::remove_const<
	    	typename HYDRA_EXTERNAL_NS::thrust::detail::remove_reference<T>::type
		>::type>::value), return_type>::type
	interface(T&& x)  const
	{
		//fNArgs=0;
		return static_cast<const Functor*>(this)->Evaluate(x);
	}


    int fCacheIndex;
	bool fCached;
    GReal_t fNorm;

protected:

    BaseFunctor<Functor, ReturnType, NPARAM>& _par;

};



}//namespace hydra


#endif /* FUNCTION_H_ */
