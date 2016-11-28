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

/**
 * \file
 * \ingroup fit
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
#include <pair>
#include <initializer_list>
#include <memory>
#include <unordered_map>

namespace hydra
{

namespace detail {

template< typename FUNCTOR, typename INTEGRATOR, size_t N>
struct PdfBase: std::enable_if< detail::is_hydra_functor<FUNCTOR>::value &&
detail::is_hydra_numerical_integrator<INTEGRATOR>::value &&(N>0)>{};


}// namespace detail


/**
 * \brief Class describing probability density functions.
 * In Hydra, a PDF has 3 componontes
 *
 *  1. not normalized functor, describing the shape
 *  2. integration algorithm, normalizes the functor
 *  3. volume of integration
 */

template<typename FUNCTOR, typename INTEGRATOR, size_t N>
struct Pdf:detail::PdfBase<FUNCTOR, INTEGRATOR,N>
{
	//tag
	typedef void hydra_pdf_tag;

	//this typedef is actually a check. If the Pdf is not built with
	//hydra::functor, PdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::PdfBase<FUNCTOR, INTEGRATOR, N>::type base_type;


	Pdf(FUNCTOR const& functor,  INTEGRATOR const& integrator,
			std::array<GReal_t,N> const& xlower,
			std::array<GReal_t,N> const& xupper,
			size_t calls = 10000):
	fIntegrator(integrator),
	fFunctor(functor),
	fXLow(xlower),
	fXUp(xupper),
	fCalls(calls),
	fNormCache(std::unordered_map<size_t, std::pair<GReal_t, GReal_t>>() )
	{
		std::tie(fNorm, fNormError) = fIntegrator(fFunctor, fXLow, fXUp, fCalls) ;

	}


	Pdf(Pdf<FUNCTOR,INTEGRATOR,N> const& other):
		fIntegrator(other.GetIntegrator()),
		fFunctor(other.GetFunctor()),
		fXLow(other.GetXLow() ),
		fXUp(other.GetXUp() ),
		fNorm(other.GetNorm() ),
		fNormError(other.GetNormError() ),
		fCalls(other.GetCalls() ),
		fNormCache(other.GetNormCache())
	{

	}


	inline Pdf<FUNCTOR,INTEGRATOR,N>&
	operator=(Pdf<FUNCTOR, INTEGRATOR,N> const & other )
	{
		if(this == &other) return *this;

		this->fXLow  = other.GetXLow() ;
		this->fXUp   = other.GetXUp() ;
		this->fNorm  = other.GetNorm() ;
		this->fNormError  = other.GetNormError() ;
		this->fCalls = other.GetCalls() ;
		this->fFunctor    = other.GetFunctor();
		this->fIntegrator = other.GetIntegrator();
		this->fNormCache  = other.GetNormCache();
		return *this;
	}


	inline	void PrintRegisteredParameters()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		fFunctor.PrintRegisteredParameters();
		HYDRA_MSG <<"Registered parameters end."<< HYDRA_ENDL;
		HYDRA_MSG << HYDRA_ENDL;
	}

	inline	void SetParameters(const std::vector<double>& parameters){

		fFunctor.SetParameters(parameters);

		size_t key = detail::hash_range(parameters.begin(),
						parameters.end());

		auto search = fNormCache.find(key);
		if (search != fNormCache.end() && fNormCache.size()>0) {

			std::tie(fNorm, fNormError) = *(search);
		}
		else {

			std::tie(fNorm, fNormError) =  fIntegrator(fFunctor, fXLow, fXUp, fCalls) ;
			fNormCache[key] = std::make_pair(fNorm, fNormError);
		}


		fFunctor.SetNorm(fNorm);
		return;
	}


	inline	INTEGRATOR& GetIntegrator() {
		return fIntegrator;
	}

	inline	INTEGRATOR& GetIntegrator() const {
		return fIntegrator;
	}

	inline	FUNCTOR& GetFunctor() const {
		return fFunctor;
	}


	 inline GReal_t GetNorm() const {
		return fNorm;
	}

	inline	void SetNorm( )
	{
		fNorm = thrust::get<0>( fIntegrator(fFunctor, fXLow, fXUp, fCalls) );
	}

	size_t GetCalls() const
	{
		return fCalls;
	}

	void SetCalls(size_t calls)
	{
		fCalls = calls;
	}


	std::array<GReal_t,N> GetXLow() const
	{
		return fXLow;
	}

	void SetXLow(std::array<GReal_t,N> const& xLow)
	{
		fXLow = xLow;
	}

	std::array<GReal_t,N> GetXUp() const
	{
		return fXUp;
	}

	void SetXUp(std::array<GReal_t,N> const& xUp)
	{
		fXUp = xUp;
	}

	std::unordered_map<size_t,std::pair<GReal_t,GReal_t> >& GetNormCache() const
	{
		return fNormCache;
	}

 	template<typename T1>
  	inline  	GReal_t operator()(T1&& t )
  	{
  		return fFunctor(t)/fNorm;

  	}

  	template<typename T1, typename T2>
  	inline  	GReal_t operator()( T1&& t, T2&& cache)
  	{

  		return fFunctor(t, cache)/fNorm;
  	}

  	template<typename T>
   inline  GReal_t operator()( T* x, T* p)
  	  	{

  	  		return fFunctor(x,p)/fNorm;
  	  	}


private:


	FUNCTOR fFunctor;
	INTEGRATOR fIntegrator;
	std::array<GReal_t,N> fXLow;
	std::array<GReal_t,N> fXUp;
	mutable GReal_t fNorm;
	mutable GReal_t fNormError;
	mutable std::unordered_map<size_t, std::pair<GReal_t, GReal_t>> fNormCache;

	size_t fCalls;
};

//get pdf from functor expression

template<typename FUNCTOR, typename INTEGRATOR, size_t N>
Pdf<FUNCTOR, INTEGRATOR, N > make_pdf( FUNCTOR const& functor,  INTEGRATOR integrator,
		std::array<GReal_t,N> const& xlower,	std::array<GReal_t,N> const& xupper){

	return Pdf<FUNCTOR, INTEGRATOR, N >(functor, integrator,  xlower, xupper);
}

}//namespace hydra




#endif /* PDF_H_ */
