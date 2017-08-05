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
#include <hydra/detail/Integrator.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/FunctorTraits.h>
//#include <hydra/UserParameters.h>

#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
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
struct BaseFunctor
{
	//tag
    typedef void hydra_functor_tag;
	typedef   ReturnType return_type;
	typedef   std::true_type is_functor;

	static const size_t parameter_count =NPARAM;


	/**
	 * Default constructor
	 */
	explicit BaseFunctor():
		fCacheIndex(-1),
		fCached(0),
		fNArgs(-1),
		fParamResgistered(1),
		fNorm(1.0),
		fNormalized(1),
		_par(*this)
	{}


    /**
     * @brief Constructor taking a list of parameters
     * @param init_parameters std::initializer_list<Parameter> with the parameters of the functor.
     */
	BaseFunctor(std::initializer_list<Parameter> init_parameters):
	fCacheIndex(-1),
	fCached(0),
	fNArgs(-1),
	fParamResgistered(1),
	fNorm(1.0),
	fNormalized(1),
	_par(*this)
	{
		if(NPARAM!=0){
		for(unsigned int i=0; i<NPARAM; i++)
			this->SetParameter(i, *(init_parameters.begin() + i));}
	}

	/**
	 * @brief Constructor taking std::array of parameters
	 * @param init_parameters std::array<Parameter,NPARAM> with the parameters of the functor.
	 */
	BaseFunctor(std::array<Parameter,NPARAM> const& init_parameters):
		fCacheIndex(-1),
		fCached(0),
		fNArgs(-1),
		fParamResgistered(1),
		fNorm(1.0),
		fNormalized(1),
		_par(*this)
		{
		if(NPARAM!=0){
			for(unsigned int i=0; i<NPARAM; i++)
				this->SetParameter(i, *(init_parameters.begin() + i));
		}
		}


	/**
	 * @brief Copy constructor
	 */
	__host__ __device__
	BaseFunctor(BaseFunctor<Functor,ReturnType, NPARAM> const& other):
	fCacheIndex( other.GetCacheIndex() ),
	fCached( other.IsCached() ),
	fNArgs(other.GetNArgs()),
	fParamResgistered(1),
	fNorm(other.GetNorm()),
	fNormalized(other.GetNormalized() ),
	_par(*this)
	{
if(NPARAM!=0){
	for(unsigned int i=0; i<NPARAM; i++)
	this->SetParameter(i, other.GetParameter(i));
}
	}

	/**
	 * @brief Assignment operator
	 */
	__host__ __device__ inline
	BaseFunctor<Functor,ReturnType, NPARAM>&
	operator=(BaseFunctor<Functor, ReturnType, NPARAM> const & other )
	{
		if(this != &other)
		{
			this->fCacheIndex     = other.GetCacheIndex();
			this->fCached         = other.IsCached();
			//this->fParameterIndex = other.GetParameterIndex();
			this->fNorm = other.GetNorm();
			this->fNormalized =other.GetNormalized();
			this->fParamResgistered =1;
			this->fNArgs= other.GetNArgs();
			if(NPARAM!=0){
			for(unsigned int i=0; i<NPARAM; i++)
				this->SetParameter(i, other.GetParameter(i));
			}
			_par=*this;

		}
		return *this;
	}


	__host__ __device__ inline
	Functor& GetFunctor() {return *static_cast<Functor*>(this);}

	__host__ __device__ inline
	int GetCacheIndex() const { return this->fCacheIndex; }

	__host__ __device__ inline
	void SetCacheIndex(int index) {fCacheIndex = index;}

	__host__ __device__ inline
	bool IsCached() const
	{ return this->fCached;}

	__host__ __device__ inline
	void SetCached(bool cached=true)
	{ fCached = cached; }


	/**
	 * @brief Print registered parameters.
	 */
	void PrintRegisteredParameters()
	{
		if(!fParamResgistered){
			HYDRA_LOG(WARNING, "Parameters not registered, check client implementation. Nothing to dump. Exiting..." )
		return;
		}

		HYDRA_CALLER ;
		HYDRA_MSG <<HYDRA_ENDL;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		for(size_t i=0; i<parameter_count; i++ )
		HYDRA_MSG <<"  >> Parameter " << i <<") "<< fParameters[i] << HYDRA_ENDL;
		HYDRA_MSG <<"Normalization " << fNorm << HYDRA_ENDL;
		HYDRA_MSG <<"Registered parameters end." << HYDRA_ENDL;
		HYDRA_MSG <<HYDRA_ENDL;
		return;
	}


	/**
	 * @brief Set parameters
	 * @param parameters
	 */
	__host__ inline
	void SetParameters(const std::vector<double>& parameters)
	{

		if(fCached) return;
		if(!fParamResgistered){
					HYDRA_LOG(WARNING, "Parameters not registered, check client implementation. Nothing to dump. Exiting..." )
		            return;
				}
		for(size_t i=0; i< parameter_count; i++){

			fParameters[i] = parameters[fParameters[i].GetIndex()];
		}

		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameter_count; i++){
			     stringStream << "Parameter["<< fParameters[i].GetIndex() <<"] :  "
			    		 << parameters[fParameters[i].GetIndex() ] << "  " << fParameters[i] << "\n";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		return;
	}


	inline	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
	{

		for(size_t i=0; i<NPARAM; i++)
			user_parameters.push_back(&fParameters[i]);
	}

	__host__ __device__ inline
		constexpr  size_t GetNumberOfParameters() const {
				return NPARAM;
	}

	__host__ __device__ inline
	const Parameter* GetParameters() const {
			return &fParameters[0];
	}

	__host__ __device__ inline
	const Parameter& GetParameter(size_t i) const {
		return fParameters[i];
	}

	__host__ __device__ inline
	void SetParameter(size_t i, Parameter const& value) {
		fParameters[i]=value;
	}

	__host__ __device__ inline
	void SetParameter(size_t i, double value) {
			fParameters[i]=value;
		}


	__host__ __device__  inline
	GReal_t GetNorm() const {
		return fNorm;
	}

	__host__ __device__  inline
	void SetNorm(GReal_t norm) {
		fNorm = norm;
	}

	__host__ __device__  inline
	void SetNormalized( bool flag ) {
		fNormalized = flag;
	}

	__host__ __device__  inline
	bool GetNormalized(  ) const {
		return fNormalized;
	}


	template<typename T>
	__host__ __device__ inline
	typename thrust::detail::enable_if<
	! ( detail::is_instantiation_of<thrust::tuple,typename std::remove_reference<T>::type >::value ||
	  detail::is_instantiation_of<thrust::detail::tuple_of_iterator_references, typename std::remove_reference<T>::type >::value )
	, return_type>::type
	interface(T&& x)
	{
		fNArgs=1;
		return static_cast<Functor*>(this)->Evaluate(&x);
	}


	template<typename T>
	__host__ __device__ inline
	typename thrust::detail::enable_if<( detail::is_instantiation_of<thrust::tuple,typename std::remove_reference<T>::type >::value ||
			  detail::is_instantiation_of<thrust::detail::tuple_of_iterator_references,typename std::remove_reference<T>::type >::value ) ||
	detail::is_homogeneous<	typename thrust::tuple_element<0, typename std::remove_reference<T>::type>::type,
	typename std::remove_reference<T>::type>::value, return_type>::type
	interface(T&& x)
	{
		typedef typename std::remove_reference<T>::type Tprime;
		typedef typename std::remove_reference<typename thrust::tuple_element<0, Tprime>::type>::type first_type;
		constexpr size_t N = thrust::tuple_size< Tprime >::value;

		first_type Array[ N ];

		detail::tupleToArray(x, &Array[0] );
		fNArgs=N;
		return static_cast<Functor*>(this)->Evaluate(&Array[0]);


	}

	template<typename T >
	__host__ __device__ inline
	typename thrust::detail::enable_if<!(detail::is_homogeneous<
	typename thrust::tuple_element<0, typename std::remove_reference<T>::type>::type,
	typename std::remove_reference<T>::type>::value), return_type>::type
	interface(T&& x)
	{
		fNArgs=0;
		return static_cast<Functor*>(this)->Evaluate(x);
	}



	template<typename T>
	__host__  __device__ inline
	return_type operator()(T* x)
	{
		GReal_t norm = fNormalized? fNorm : 1.0;

		return norm>0.0?static_cast<Functor*>(this)->Evaluate(x)*norm:0;
	}


	template<typename T>
	__host__ __device__ inline
	return_type operator()( T&&  x )
	{
		GReal_t norm = fNormalized ? fNorm : 1.0;

		return  norm>0.0? interface( x)*norm: 0;
	}


	template<typename T1, typename T2>
	__host__ __device__  inline
	return_type operator()( T1&& x, T2&& cache)
	{

		GReal_t norm = fNormalized? fNorm : 1.0;


		return fCached ?\
				detail::extract<return_type, T2 >(fCacheIndex, std::forward<T2>(cache)):\
				norm>0.0?operator()<T1>( std::forward<T1>(x) )*norm:0;
	}



    __host__ __device__  inline
	GReal_t operator[](unsigned int i) const
	{
		return (GReal_t ) fParameters[i];
	}

    __host__ __device__  inline
	int GetNArgs() const {
		return fNArgs;
	}

private:

    int fCacheIndex;
	bool fCached;
	int fNArgs;
	bool fParamResgistered;
    GReal_t fNorm;
	bool fNormalized;
	Parameter fParameters[NPARAM];

protected:

    BaseFunctor<Functor, ReturnType, NPARAM>& _par;

};



}//namespace hydra


#endif /* FUNCTION_H_ */
