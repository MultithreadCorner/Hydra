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
 * LogLikelihoodFCN.h
 *
 *  Created on: 03/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup fit
 */

#ifndef _LOGLIKELIHOODFCN_H_
#define _LOGLIKELIHOODFCN_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/FCN.h>
#include <hydra/detail/functors/LogLikelihood.h>
#include <hydra/PointVector.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/Print.h>
#include <hydra/UserParameters.h>

#include <thrust/distance.h>
#include <thrust/tuple.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>

#include <Minuit2/FCNBase.h>
#include <vector>
#include <unordered_map>
#include <cassert>


namespace hydra{

/**
 * \ingroup fit
 * \brief FCN for  maximum likelihood fits.
 *
 * This class compute the ROOT::Minuit2 FCN function for maximum likelihood fits.
 * An instance of this class can be passed directly to the  ROOT::Minuit2 interface.
 * Very rarely it is necessary to instantiate this class template directly.
 * Instead, given a hydra::PDF and multidimensional dataset stored in a hydra::PointVector< PointType, BACKEND>, the user can use
 * the hydra::make_fcn function to get a  suitable instance.
 *
 * ##Example##
 \code{.cpp}

//std
#include <iostream>
#include <string>
#include <vector>
#include <array>

//hydra
#include <hydra/Random.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinimize.h"


#include <examples/Gauss.h>
#include <examples/Exp.h>

using namespace std;
using namespace ROOT::Minuit2;
using namespace hydra;
using namespace examples;
 ...

    //Generator with current time count as seed.
	size_t nentries = 10e6;


	//-------------------------------------------
	//range of the analysis
	std::array<double, 1>  min   ={ 0.0};
	std::array<double, 1>  max   ={ 15.0};

	//------------------------------------
	//parameters names
	std::string Mean("Mean"); 		// mean of the Gaussian
	std::string Sigma("Sigma"); 	// sigma of the Gaussian
	std::string Tau("Tau"); 		// tau of Exponential
	std::string Yield_Gauss("Yield_Gauss"); // yield of Gaussian component
	std::string Yield_Exp("Yield_Exp"); 	// yield of Exponential component

	//fit paremeters
	//----------------------------------------------------------------------
	// create parameters
	//	Gaussian:
	//	mean = 3.0,	sigma = 0.5, yield = N1_p
	//
	//	Exponential:
	//	tau  = 1.0

	// 1) using named parameter idiom
	Parameter  mean_p   = Parameter::Create()
								 .Name(Mean)
								 .Value(3.0)
								 .Error(0.000)
								 .Limits(1.0, 4.0)
								 ;

	Parameter  sigma_p  = Parameter::Create()
									.Name(Sigma)
									.Value(0.5)
									.Error(0.0001)
									.Limits(0.1, 1.5)
									;

	Parameter  tau_p    = Parameter::Create()
									.Name(Tau)
									.Value(0.0014)
									.Error(0.0001)
									.Limits( -1.0, 1.0)
									;

	// 2) using unnamed parameter idiom
	Parameter NYield_Gauss_p(Yield_Gauss ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;
	Parameter NYield_Exp_p(Yield_Exp     ,nentries, sqrt(nentries), nentries-nentries/2 , nentries+nentries/2) ;


    // Setting the verbosity of Minuit2
	ROOT::Minuit2::MnPrint::SetLevel(3);


	// create functors
	Gauss Gaussian1(mean_p, sigma_p,0);
	Exp   Exponential(tau_p,0);

    //Generate data on the device with the original parameters
	typedef Point<double, 1> point_t;
	PointVector<point_t , hydra::device::sys> data_d(2*nentries);

	size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	Random<thrust::random::default_random_engine> Generator( seed  );

	Generator.Gauss(mean_p , sigma_p,
			GetCoordinateBegin<0>(data_d),
			GetCoordinateBegin<0>(data_d) + nentries );

	Generator.Uniform(min[0], max[0],
			GetCoordinateBegin<0>(data_d)+ nentries,
			GetCoordinateEnd<0>(data_d) );

	//perform the fit on the device
	{

		GaussAnalyticIntegral GaussIntegral(min[0], max[0]);
		ExpAnalyticIntegral   ExpIntegral(min[0], max[0]);

		auto Gaussian_PDF    = make_pdf(Gaussian, GaussIntegral);
		auto Exponential_PDF = make_pdf(Exponential, ExpIntegral);

		//----------------------------------------------------------------------
		//add the pds to make a extended pdf model

		//list of yields
		std::array<Parameter*, 2>  yields{&NYield_Gauss_p, &NYield_Exp_p};

		auto model = add_pdfs(yields, Gaussian_PDF, Exponential_PDF );
		model.SetExtended(1);

		//-------------------------------------------------
		//minimization

		//get the FCN
		auto ModelFCN_d = make_loglikehood_fcn(model, data_d);

		//print Minuit parameters before the fit
		std::cout << ModelFCN_d.GetParameters().GetMnState() << endl;

        //print the model before the fit
		ModelFCN_d.GetPDF().PrintRegisteredParameters();

		//minimization strategy
		MnStrategy strategy(2);

		// create Migrad minimizer
		MnMigrad migrad_d(modelFCN_d, modelFCN_d.GetParameters().GetMnState() ,  strategy);

		FunctionMinimum *minimum_d=0;
		FunctionMinimum *minimum_h=0;

		// ... Minimize and profile the time
		minimum_d = new FunctionMinimum( migrad_d(5000, tolerance));

		// output the fit result
		std::cout<<"minimum: "<<*minimum_d<<std::endl;

        delete minimum_d;
	}

	...
 \endcode

 *[*To find basic information about Maximum likelihood estimation look at Wikipedia page and references there in.*](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation)
*/


template<typename PDF, typename PointType, typename IteratorData,typename IteratorCache >
class LogLikelihoodFCN: public FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>
{

public:


	/**
	 * \brief LogLikelihoodFCN constructor for non-cached models.
	 *
	 * @param functor hydra::PDF instance.
	 * @param begin  iterator pointing to the begin of the dataset.
	 * @param end   iterator pointing to the end of the dataset.
	 */
	LogLikelihoodFCN(PDF& functor, IteratorData begin, IteratorData end):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(functor,begin, end),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
		{}

	/**
	 * \brief LogLikelihoodFCN constructor for cached models.
	 * @param functor hydra::PDF instance.
	 * @param begin iterator pointing to the begin of the dataset.
	 * @param end iterator pointing to the end of the dataset.
	 * @param cend iterator pointing to the begin of the cache table.
	 */
	LogLikelihoodFCN(PDF& functor, IteratorData begin, IteratorData end, IteratorCache cbegin):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(functor,begin, end, cbegin),
		fMAxValue(std::numeric_limits<GReal_t>::min() )
		{}

	/**
	 *\brief  Copy constructor.
	 * @param other
	 */
	LogLikelihoodFCN( LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache> const& other):
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>(other),
		fMAxValue(other.GetMAxValue())
		{}

	/**
	 * \brief  Assignment operator.
	 * @param other LogLikelihoodFCN instance.
	 * @return a
	 */
	LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>&
	operator=(LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache> const& other)
	{
		FCN<LogLikelihoodFCN<PDF, PointType, IteratorData, IteratorCache>>::operator=(other);
		this->fMAxValue=other.GetMAxValue();
		return *this;
	}

	/**
	 * \brief Evaluate the Log-likelihood for a model consisting of only one hydra::PDF.
	 * @param parameters are the set of parameter values passed by Minuit2.
	 * @return a double corresponding to the value of the log-likelihood for the set of parameters.
	 */
	template<typename U= PDF>
	typename thrust::detail::enable_if< hydra::detail::is_hydra_pdf<U>::value, GReal_t>::type
	Eval( const std::vector<double>& parameters ) const
	{
		using thrust::system::detail::generic::select_system;
		typedef typename thrust::iterator_system<IteratorData>::type System;
		System system;

		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + this->GetNEvents();

		GReal_t final;
		GReal_t init=0;

		//set functor with the new parameters
		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		//std::cout << "Mean1 " << parameters[0] << std::endl;
		this->GetPDF().SetParameters(parameters);
		this->GetPDF().PrintRegisteredParameters();


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<typename U::functor_type, PointType,
				IteratorData, IteratorCache>( this->GetPDF().GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());


		//std::cout <<this->GetSumW() + final << std::endl;
		return this->GetSumW() + final ;
	}

	/**
	 * \brief Evaluate the Log-likelihood for a model consisting of a sum of hydra::PDF.
	 * @param parameters are the set of parameter values passed by Minuit2.
	 * @return a double corresponding to the value of the log-likelihood for the set of parameters.
	 */
	template<typename U=PDF>
	typename thrust::detail::enable_if< hydra::detail::is_hydra_sum_pdf<U>::value, GReal_t>::type
     Eval( const std::vector<double>& parameters ) const
	{

		using thrust::system::detail::generic::select_system;
		typedef typename thrust::iterator_system<IteratorData>::type System;
		System system;

		// create iterators
		thrust::counting_iterator<size_t> first(0);
		thrust::counting_iterator<size_t> last = first + this->GetNEvents();

		GReal_t final;
		GReal_t init=0;

		//set functor with the new parameters
		if (INFO >= hydra::Print::Level()  )
		{
			std::ostringstream stringStream;
			for(size_t i=0; i< parameters.size(); i++){
				stringStream << "Parameter["<< i<<"] :  " << parameters[i]  << "  ";
			}
			HYDRA_LOG(INFO, stringStream.str().c_str() )
		}

		this->GetPDF().SetParameters(parameters);

		/*
		 * avoid evaluate inconsistent coefficients configurations
		 * returning quickly the highest NLL value already calculated
		 */
		if( this->GetPDF().IsFractioned() &&
				( this->GetPDF().GetCoefSum() < 0.0 || this->GetPDF().GetCoefSum() > 0.0) )
			return fMAxValue;


		final=thrust::transform_reduce(select_system(system), first, last,
				detail::LogLikelihood<typename U::functor_type, PointType,
				IteratorData, IteratorCache>( this->GetPDF().GetFunctor(),
						this->GetSumW()     , this->GetSumW2() ,
						this->GetDataBegin(), this->GetCacheBegin(),
						this->isWeighted()  , this->isCached()),
						init, thrust::plus<GReal_t>());

		GReal_t  r = this->GetSumW() + this->GetPDF().IsExtended()*
				( this->GetPDF().GetCoefSum() -
						this->GetSumW()*log(this->GetPDF().GetCoefSum() ) ) - final;

		fMAxValue = fMAxValue<r?r:fMAxValue;
	//	std::cout << std::setprecision(16)<<r << std::endl;

		return r;
	}


    /**
     * \brief Get the maximum value registered during the minimization/
     * @return a double.
     */
	GReal_t GetMAxValue() const {
		return fMAxValue;
	}

private:

	mutable GReal_t fMAxValue;


};

/**
 * \brief Function for build a hydra::LogLikelihoodFCN from a non-cached hydra::PDF and a
 *  hydra::PointVector dataset.
 * @param functor hydra::PDF.
 * @param data hydra::PointVector dataset.
 * @return a hydra::LogLikelihoodFCN object.
 */
template<typename PDF, typename PointType, unsigned int BACKEND>
auto make_loglikehood_fcn(PDF& functor,
		PointVector< PointType, BACKEND> const& data)
-> LogLikelihoodFCN<PDF, PointType,
typename PointVector< PointType, BACKEND>::const_iterator,
thrust::constant_iterator<null_type> >
{
	return LogLikelihoodFCN<PDF, PointType,
			typename PointVector< PointType, BACKEND>::const_iterator,
			thrust::constant_iterator<null_type>>(functor,  data.cbegin(),  data.cend());
}

/**
 * \brief Function for build a hydra::LogLikelihoodFCN from a cached hydra::PDF and a
 *  hydra::PointVector dataset.
 * @param functor hydra::PDF.
 * @param data hydra::PointVector dataset.
 * @param cache table stored as a multivector.
 * @return a hydra::LogLikelihoodFCN object.
 */
template<typename PDF,  typename PointType,	template<typename...> class Vector,
template<typename...> class Allocator, typename Tuple,  unsigned int BACKEND>
auto make_loglikehood_fcn(PDF& functor,
		PointVector< PointType, BACKEND> const& data,
		multivector<Vector<Tuple, Allocator< Tuple >>> const& cache )
-> LogLikelihoodFCN<PDF,  PointType, typename PointVector< PointType, BACKEND>::iterator,
typename multivector<Vector<Tuple, Allocator< Tuple >>>::iterator >
{
	return LogLikelihoodFCN<PDF, PointType,
			typename PointVector<PointType,BACKEND>::iterator,
			typename multivector<Vector<Tuple, Allocator< Tuple >>>::iterator >( functor, data.cbegin(), data.cend(), cache.cbegin());
}


}//namespace hydra


#endif /* LOGLIKELIHOODFCN_H_ */
