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
#include <utility>
#include <initializer_list>
#include <memory>
#include <unordered_map>


namespace hydra
{

namespace detail {

template< typename FUNCTOR, typename INTEGRATOR>
struct PdfBase: std::enable_if< detail::is_hydra_functor<FUNCTOR>::value &&
detail::is_hydra_integrator<INTEGRATOR>::value>{

	typedef FUNCTOR functor_type;

};


}// namespace detail


/**
 * \brief Class describing probability density functions.
 * In Hydra, a PDF has 3 componontes
 *
 *  1. not normalized functor, describing the shape
 *  2. integration algorithm, normalizes the functor
 *  3. volume of integration
 */

template<typename FUNCTOR, typename INTEGRATOR>
struct Pdf:detail::PdfBase<FUNCTOR, INTEGRATOR>
{
	//tag
	typedef void hydra_pdf_tag;

	//this typedef is actually a check. If the Pdf is not built with
	//hydra::functor, PdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::PdfBase<FUNCTOR, INTEGRATOR>::type base_type;



	Pdf(FUNCTOR const& functor,  INTEGRATOR const& integrator):
	fIntegrator(integrator),
	fFunctor(functor),
	fNormCache(std::unordered_map<size_t, std::pair<GReal_t, GReal_t>>() )
	{
		std::tie(fNorm, fNormError) = fIntegrator(fFunctor) ;
		fFunctor.SetNorm(1.0/fNorm);
		for(size_t i=0; i< FUNCTOR::parameter_count; i++)
			fParameters[i]=fFunctor.GetParameter(i);
		size_t key = detail::hash_range(fParameters.begin(),
						fParameters.end());

		fNormCache[key] = std::make_pair(fNorm, fNormError);

	}


	Pdf(Pdf<FUNCTOR,INTEGRATOR> const& other):
		fIntegrator(other.GetIntegrator()),
		fFunctor(other.GetFunctor()),
		fNorm(other.GetNorm() ),
		fNormError(other.GetNormError() ),
		fNormCache(other.GetNormCache())
	{

		for(size_t i=0; i< FUNCTOR::parameter_count; i++)
			fParameters[i]=fFunctor.GetParameter(i);

		size_t key = detail::hash_range(fParameters.begin(),
				fParameters.end());

		fNormCache[key] = std::make_pair(fNorm, fNormError);
		//fFunctor.SetNorm(1.0/fNorm);
	}


	inline Pdf<FUNCTOR,INTEGRATOR>&
	operator=(Pdf<FUNCTOR, INTEGRATOR> const & other )
	{
		if(this == &other) return *this;

		this->fNorm  = other.GetNorm() ;
		this->fNormError  = other.GetNormError() ;
		this->fFunctor    = other.GetFunctor();
		this->fIntegrator = other.GetIntegrator();
		this->fNormCache  = other.GetNormCache();
		for(size_t i=0; i< FUNCTOR::parameter_count; i++)
			this->fParameters[i]=fFunctor.GetParameter(i);


		size_t key = detail::hash_range(fParameters.begin(),
				fParameters.end());

		fNormCache[key] = std::make_pair(fNorm, fNormError);

		//fFunctor.SetNorm(1.0/fNorm);
		return *this;
	}

	~Pdf(){}

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

		for(size_t i=0; i< FUNCTOR::parameter_count; i++){
			fParameters[i]=fFunctor.GetParameter(i);
			}


		size_t key = detail::hash_range(fParameters.begin(),
				fParameters.end());


		auto search = fNormCache.find(key);
		if (search != fNormCache.end() && fNormCache.size()>0) {

			std::tie(fNorm, fNormError) = search->second;
			//std::cout << "found norm! key="<<key << std::endl;

		}
		else {

			std::tie(fNorm, fNormError) =  fIntegrator(fFunctor) ;
			fNormCache[key] = std::make_pair(fNorm, fNormError);

		}

		fFunctor.SetNorm(1.0/fNorm);

		return;
	}


	inline	INTEGRATOR& GetIntegrator() {
		return fIntegrator;
	}

	inline  const 	INTEGRATOR& GetIntegrator() const {
		return fIntegrator;
	}

	inline	const FUNCTOR& GetFunctor() const {
		return fFunctor;
	}

	inline	FUNCTOR& GetFunctor() {
			return fFunctor;
		}


	inline GReal_t GetNorm() const {
		return fNorm;
	}

	inline GReal_t GetNormError() const {
			return fNormError;
	}

	inline	void Normalize( )
	{
		std::tie(fNorm, fNormError )  =  fIntegrator(fFunctor) ;

	}



	std::unordered_map<size_t,std::pair<GReal_t,GReal_t> >& GetNormCache() const
	{
		return fNormCache;
	}

 	template<typename T1>
  	inline  	GReal_t operator()(T1&& t )
  	{
  		return fFunctor(t);

  	}

  	template<typename T1, typename T2>
  	inline  	GReal_t operator()( T1&& t, T2&& cache)
  	{

  		return fFunctor(t, cache);
  	}

  	template<typename T>
   inline  GReal_t operator()( T* x, T* p)
  	  	{

  	  		return fFunctor(x,p);
  	  	}


private:

  	std::array<GReal_t, FUNCTOR::parameter_count> fParameters;
  	FUNCTOR fFunctor;
  	INTEGRATOR fIntegrator;
	GReal_t fNorm;
	GReal_t fNormError;
	mutable std::unordered_map<size_t, std::pair<GReal_t, GReal_t>> fNormCache;

};

//get pdf from functor expression

template<typename FUNCTOR, typename INTEGRATOR>
Pdf<FUNCTOR, INTEGRATOR> make_pdf( FUNCTOR const& functor,  INTEGRATOR integrator){

	return Pdf<FUNCTOR, INTEGRATOR>(functor, integrator);
}

}//namespace hydra




#endif /* PDF_H_ */
