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
 * Pdf.h
 *
 *  Created on: 25/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PDF_H_
#define PDF_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/Parameter.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/FunctorTraits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <array>
#include <initializer_list>
#include <memory>
#include <unordered_map>

namespace hydra
{

namespace detail {

template<typename FUNCTOR, typename INTEGRATOR>
struct PdfBase: std::enable_if< detail::is_hydra_functor<FUNCTOR>::value &&
detail::is_hydra_integrator<INTEGRATOR>::value>{};


}// namespace detail

template<typename FUNCTOR, typename INTEGRATOR>
struct Pdf:detail::PdfBase<FUNCTOR, INTEGRATOR>
{
	//tag
	typedef void hydra_pdf_tag;

	//this typedef is actually a check. If the Pdf is not built with
	//hydra::functor, PdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::PdfBase<FUNCTOR, INTEGRATOR>::type base_type;



	template<typename U=FUNCTOR>
	__host__ __device__
	Pdf(U const& functor,  INTEGRATOR* integrator):
	fIntegrator(integrator),
	fFunctor(functor),
	fNorm(1.0)
	{
/*
#ifndef __CUDA_ARCH__
		fNormCache = norm_cache_type(new std::unordered_map<size_t, GReal_t>());
		//fIntegrator->Integrate(functor, 1);
		//fNorm=fIntegrator->GetResult();

#endif
*/
	}

	//__hydra_exec_check_disable__
	__host__ __device__
	Pdf(Pdf<FUNCTOR,INTEGRATOR> const& other):
	fIntegrator(other.GetIntegrator()),
	fFunctor(other.GetFunctor()),
	fNorm(other.GetNorm())
	{
/*
#ifndef __CUDA_ARCH__
		fNormCache= other.GetNormCache();
#endif
*/
	}


	__host__ __device__
	~Pdf(){	}

	__host__ __device__ inline
	Pdf<FUNCTOR,INTEGRATOR>&
	operator=(Pdf<FUNCTOR, INTEGRATOR> const & other )
	{
		if(this == &other) return *this;

		this->fNorm = other.GetNorm();
		this->fFunctor=other.GetFunctor();
		this->fIntegrator = other.GetIntegrator();
		/*
#ifndef __CUDA_ARCH__
		this->fNormCache=other.GetNormCache();
#endif
*/
		return *this;
	}

/*
	norm_cache_type GetNormCache() const {
		return fNormCache;
	}

	__host__ inline
	void SetNormCache(norm_cache_type normCache) {
		fNormCache = normCache;
	}
*/

	__host__ inline
	void PrintRegisteredParameters()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		fFunctor.PrintRegisteredParameters();
		HYDRA_MSG <<"Registered parameters end."<< HYDRA_ENDL;
		HYDRA_MSG << HYDRA_ENDL;
	}

	__host__ inline
	void SetParameters(const std::vector<double>& parameters){

		fFunctor.SetParameters(parameters);
		UpdateNorm(parameters);

		return;
	}


	__host__ inline
	void UpdateNorm( const std::vector<double>& parameters ) const
	{

		GReal_t value = EvalIntegral();
		fNorm = value;
		if (INFO >= hydra::Print::Level()){
			std::ostringstream stringStream;
			stringStream << "Setting fNorm to value " << fNorm ;
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}


	}
	__host__ __device__ inline
	INTEGRATOR* GetIntegrator() const {
		return fIntegrator;
	}

	__host__ inline
	void SetIntegrator(INTEGRATOR* integrator) {
		fIntegrator = integrator;
	}
	__host__ __device__ inline
	FUNCTOR GetFunctor() const {
		return fFunctor;
	}
	__host__ __device__ inline
	void SetFunctor(FUNCTOR functor) {
		fFunctor = functor;
	}
	__host__ __device__ inline
	GReal_t GetNorm() const {
		return fNorm;
	}
	__host__ __device__ inline
	void SetNorm( GReal_t norm) {
		fNorm = norm;
	}

 	template<typename T1>
  	__host__ __device__ inline
  	GReal_t operator()(T1&& t )
  	{
  		return fFunctor(t)/fNorm;

  	}

  	template<typename T1, typename T2>
  	__host__ __device__  inline
  	GReal_t operator()( T1&& t, T2&& cache)
  	{

  		return fFunctor(t, cache)/fNorm;
  	}

  	template<typename T>
  	  	__host__ __device__ inline
  	  GReal_t operator()( T* x, T* p)
  	  	{

  	  		return fFunctor(x,p)/fNorm;
  	  	}


private:

	__host__ inline
	GReal_t EvalIntegral( ) const
	{
		 constexpr bool has_ana_integral =  detail::has_analytical_integral<
				FUNCTOR, GReal_t(const GReal_t*,  const GReal_t*)>::value;

		if( has_ana_integral ){

			return	this->EvalAnalyticIntegral<FUNCTOR>(fIntegrator->GetLimits().first,
					fIntegrator->GetLimits().second );
		}
		else{

			return	this->EvalNumericalIntegral();
		}
	}

	__host__ inline
	GReal_t EvalNumericalIntegral( ) const {

		HYDRA_LOG(INFO, "Calculating integral numerically" )

		fIntegrator->Integrate(fFunctor, 1);
		return fIntegrator->GetResult();
	}

	template<typename U=FUNCTOR>
	__host__ inline
	typename thrust::detail::enable_if<
	detail::has_analytical_integral<U, GReal_t(const GReal_t*,  const GReal_t*)>::value, GReal_t >::type
	EvalAnalyticIntegral(const GReal_t* down,  const GReal_t* up) const{

		HYDRA_LOG(INFO, "Calculating integral analytically. Calling ClientFunctor::AnalyticalIntegral( down, up )" )
		return static_cast<U*>(this)->AnalyticalIntegral( down, up );
	}

	template<typename U=FUNCTOR>
	__host__ inline
	typename thrust::detail::enable_if<
	!detail::has_analytical_integral<U, GReal_t(const GReal_t*,  const GReal_t*)>::value, GReal_t >::type
	EvalAnalyticIntegral(const GReal_t*,  const GReal_t* ) const{
		HYDRA_LOG(INFO, "ClientFunctor::AnalyticalIntegral( down, up ) not implemented. This message shoul never show up..." )
		return 1.0;
	}

	mutable GReal_t fNorm;
	FUNCTOR fFunctor;
	INTEGRATOR* fIntegrator;
	//norm_cache_type fNormCache;
};

//get pdf from functor expression

template<typename FUNCTOR, typename INTEGRATOR>
Pdf<FUNCTOR, INTEGRATOR > make_pdf( FUNCTOR const& functor,  INTEGRATOR* integrator	)
{
	return Pdf<FUNCTOR, INTEGRATOR >(functor, integrator);
}

}//namespace hydra




#endif /* PDF_H_ */
