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
#include <utility>
#include <initializer_list>
#include <memory>
#include <unordered_map>


namespace hydra
{

namespace detail {

template< typename FUNCTOR, typename INTEGRATOR>
class PdfBase: public std::enable_if< detail::is_hydra_functor<FUNCTOR>::value &&
detail::is_hydra_integrator<INTEGRATOR>::value >
{};


}// namespace detail


/**
 * @brief Class describing probability density functions.
 * @ingroup fit
 * A hydra::Pdf has two components:
 *
 *  1. non-normalized functor, describing the shape.
 *  2. integration algorithm or functor for analytical integration, which normalizes the functor.
 */
template<typename FUNCTOR, typename INTEGRATOR>
class Pdf: public detail::PdfBase<FUNCTOR, INTEGRATOR>
{


public:
//this typedef is actually a check. If the Pdf is not built with
	//hydra::functor, PdfBase::type will not be defined and compilation
	//will fail
	typedef typename detail::PdfBase<FUNCTOR, INTEGRATOR>::type base_type;

	typedef FUNCTOR functor_type;
	//tag
	typedef void hydra_pdf_tag;


	/**
	 * @brief hydra::Pdf constructor.
	 * @param functor describing the shape.
	 * @param integrator functor for calculate analytical integrals or hydra
	 * algorithm for numerical integration.
	 */
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


	/**
	 * @brief Copy constructor.
	 * @param other
	 */
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

	~Pdf(){};

	/**
	 *@brief Assignment operator.
	 * @param other
	 * @return a hydra::Pdf equal to other.
	 */
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


	/**
	 * @brief Add pointers to the functor's parameters to a external list, that will be used later to
	 * build the hydra::UserParameters instance that will be passed to ROOT::Minuit2.
	 * @param user_parameters external std::vector<hydra::Parameter*> object holding the list of pointers
	 * to functor parameters.
	 */
	inline	void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
	{
		fFunctor.AddUserParameters(user_parameters );
	}


	/**
	 * @brief Print all registered parameters, including its value, range, name etc.
	 */
	inline	void PrintRegisteredParameters()
	{
		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:" << HYDRA_ENDL;
		fFunctor.PrintRegisteredParameters();
		HYDRA_MSG <<"Registered parameters end."<< HYDRA_ENDL;
		HYDRA_MSG << HYDRA_ENDL;
	}

	/**
	 * @brief Set the parameters of the functor to the value informed by ROOT::Minuit2.
	 * @param parameters std::vector<double> containing the list of parameters passed by ROOT::Minuit2.
	 */
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
			/*
			std::cout << ">>> Found cached norm key="<<key << std::endl;
			std::cout << ">>> Parameters values: "<< std::endl;
			for(size_t i=0; i< FUNCTOR::parameter_count; i++){
				std::cout <<">>> [" << i << "]" << std::setprecision(10)<< fParameters[i] << std::endl;
			}
			*/
		}
		else {

			std::tie(fNorm, fNormError) =  fIntegrator(fFunctor) ;
			fNormCache[key] = std::make_pair(fNorm, fNormError);

		}

		fFunctor.SetNorm(1.0/fNorm);

		return;
	}


	/**
	 * @brief Get a reference to the integrator (functor or algorithm).
	 * @return INTEGRATOR& .
	 */
	inline	INTEGRATOR& GetIntegrator() {
		return fIntegrator;
	}

	/**
	 * @brief Get a constant reference to the integrator (functor or algorithm).
	 * @return const INTEGRATOR& .
	 */
	inline  const 	INTEGRATOR& GetIntegrator() const {
		return fIntegrator;
	}

	/**
	 * @brief Get a constant reference to the functor describing the shape.
	 * @return const FUNCTOR& .
	 */
	inline	const FUNCTOR& GetFunctor() const {
		return fFunctor;
	}

	/**
	 * @brief Get a reference to the functor describing the shape.
	 * @return FUNCTOR& .
	 */
	inline	FUNCTOR& GetFunctor() {
			return fFunctor;
		}


	/**
	 * @brief Get norm of the hydra::Pdf.
	 * @return the normalization factor.
	 */
	inline GReal_t GetNorm() const {
		return fNorm;
	}

	/**
	 *  @brief Get the error on the norm of the hydra::Pdf.
	 * @return Error the normalization factor.
	 */
	inline GReal_t GetNormError() const {
			return fNormError;
	}

	/**
	 * @brief Normalize PDF.
	 *
	 */
	inline	void Normalize( )
	{
		std::tie(fNorm, fNormError )  =  fIntegrator(fFunctor) ;

	}



	/**
	 * @brief Get cache table of normalization factors.
	 * @return std::unordered_map<size_t,std::pair<GReal_t,GReal_t> > instance with the cache table.
	 */
	std::unordered_map<size_t,std::pair<GReal_t,GReal_t> >& GetNormCache() const
	{
		return fNormCache;
	}

	/**
	 * @brief Evaluate the PDF on the tuple of arguments T1.
	 * @param t Tuple of arguments.
	 * @return
	 */
 	template<typename T1>
  	inline  GReal_t operator()(T1&& t )
  	{
  		return fFunctor(t);

  	}

 	/**
 	 * @brief Evaluate the PDF on the tuple T1 using the cache table T2.
 	 * @param t Tuple of arguments.
 	 * @param cache table of pre-calculated values.
 	 * @return
 	 */
  	template<typename T1, typename T2>
  	inline  GReal_t operator()( T1&& t, T2&& cache)
  	{

  		return fFunctor(t, cache);
  	}


   template<typename T>
   inline  GReal_t operator()( T* x, T* p=0)
  	{

  	  		return fFunctor(x);
  	}


private:

  	std::array<GReal_t, FUNCTOR::parameter_count> fParameters;
  	FUNCTOR fFunctor;
  	INTEGRATOR fIntegrator;
	GReal_t fNorm;
	GReal_t fNormError;
	mutable std::unordered_map<size_t, std::pair<GReal_t, GReal_t>> fNormCache;

};


/**
 * @brief Build a hydra::Pdf given a shape described by a functor and a integrator
 *  (algorithm or functor).
 * @param functor shape.
 * @param integrator algorithm or functor.
 * @return a hydra::Pdf instance.
 */
template<typename FUNCTOR, typename INTEGRATOR>
Pdf<FUNCTOR, INTEGRATOR> make_pdf( FUNCTOR const& functor,  INTEGRATOR integrator)
{

	return Pdf<FUNCTOR, INTEGRATOR>(functor, integrator);
}

}//namespace hydra




#endif /* PDF_H_ */
