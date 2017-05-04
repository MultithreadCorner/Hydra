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
 * FCN.h
 *
 *  Created on: 14/08/2016
 *      Author: Antonio Augusto Alves Junior
 */
/**
 * \file
 * \ingroup fit
 */

#ifndef _FCN_H_
#define _FCN_H_

#include <hydra/detail/Config.h>

#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/detail/functors/LogLikelihood.h>
#include <hydra/detail/utility/Arithmetic_Tuple.h>
#include <hydra/experimental/Point.h>
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


namespace hydra {

namespace experimental {


template<typename T>
class FCN;

template<template<typename... > class ESTIMATOR, typename PDF,
typename PointType, typename IteratorData, typename IteratorCache>
class FCN<ESTIMATOR<PDF, PointType, IteratorData, IteratorCache>>: public ROOT::Minuit2::FCNBase {
public:

	typedef ESTIMATOR<PDF, PointType, IteratorData, IteratorCache> estimator_type;
	typedef PointType point_type;
	typedef typename thrust::iterator_traits<IteratorData>::value_type data_value_type;
	typedef typename thrust::iterator_traits<IteratorCache>::value_type cache_value_type;

	struct Weights { GReal_t fSumW; GReal_t fSumW2; };

	struct UnaryWeights
	{
		template<typename T>
		__host__ __device__
		Weights operator()(T data )
		{
			Weights weights;
			weights.fSumW  = thrust::get<0>(data);
			weights.fSumW2 = thrust::get<1>(data);
			return weights;
		}
	};

	struct BinaryWeights
	{
		__host__ __device__
		Weights operator()(Weights const& w1,  Weights const& w2)
		{
			Weights weights;
			weights.fSumW  = w1.fSumW + w2.fSumW;
			weights.fSumW2 = w1.fSumW2 + w2.fSumW2;
			return weights;
		}
	};


	FCN(PDF& functor, IteratorData begin, IteratorData end) :
		fPDF(functor),
		fDataBegin(begin),
		fDataEnd(end),
		fCacheBegin(IteratorCache()),
		fNEvents(thrust::distance(begin, end)),
		fErrorDef(0.5),
		fWeighted(kFalse),
		fCached(kFalse),
		fFCNCache(std::unordered_map<size_t, GReal_t>())
{
		LoadFCNParameters();

		Weights init=Weights();

		Weights  result = thrust::transform_reduce(begin, end, UnaryWeights(),
				init, BinaryWeights() );

		fSumW  =  result.fSumW;
		fSumW2 =  result.fSumW2;

		std::cout << "=======> fSumW " << fSumW <<std::endl;

}

	FCN(PDF& functor, IteratorData begin, IteratorData end, IteratorCache cbegin) :
		fPDF(functor),
		fDataBegin(begin),
		fDataEnd(end),
		fCacheBegin(cbegin),
		fNEvents(thrust::distance(begin, end)),
		fErrorDef(0.5),
		fWeighted(kFalse),
		fCached(kTrue),
		fFCNCache(std::unordered_map<size_t, GReal_t>())
	{
		//typename IteratorData::value_type init;
		LoadFCNParameters();

		Weights init= Weights();

		Weights  result = thrust::transform_reduce(begin, end, UnaryWeights(),
				init, BinaryWeights() );

		fSumW  =  result.fSumW;
		fSumW2 =  result.fSumW2;
	}

	FCN(FCN<estimator_type> const& other) :
		ROOT::Minuit2::FCNBase(other),
		fPDF(other.GetPDF()),
		fDataBegin(other.GetDataBegin()),
		fDataEnd(other.GetDataEnd()),
		fCacheBegin(other.GetCacheBegin()),
		fNEvents(other.GetNEvents()),
		fErrorDef(other.ErrorDef()),
		fWeighted(other.isWeighted()),
		fCached(other.isCached()),
		fFCNCache(other.GetFcnCache()),
		fSumW(other.GetSumW()),
		fSumW2(other.GetSumW2()),
	    fUserParameters(other.GetParameters())
	{}

	FCN<estimator_type>&
	operator=(FCN<estimator_type> const& other)
	{
		ROOT::Minuit2::FCNBase::operator = (other);
		this->fPDF=other.GetPDF();
		this->fDataBegin = other.GetDataBegin();
		this->fDataEnd = other.GetDataEnd();
		this->fCacheBegin = other.GetCacheBegin();
		this->fNEvents = other.GetNEvents();
		this->fErrorDef = other.ErrorDef();
		this->fWeighted = other.isWeighted();
		this->fCached = other.isCached();
		this->fFCNCache = other.GetFcnCache();
		this->fSumW = other.GetSumW();
		this->fSumW2 = other.GetSumW2();
		this->fUserParameters=other.GetParameters();

		return *this;
	}



	/**
	 * Up function from Minuit2
	 */
	virtual GReal_t Up() const {
		return fErrorDef;
	}

	/**
	 * GReal_t operator()(const std::vector<double>&) const
	 */
	virtual GReal_t operator()(const std::vector<double>& parameters) const {

		/*
		 * get the fcn_value corresponding to the parameters
		 * cached values are returned for revisited parameters
		 */
		GReal_t fcn_value = GetFCNValue(parameters);

		return fcn_value;

	}

	hydra::UserParameters& GetParameters()
	{

		return fUserParameters;
	}

	hydra::UserParameters const& GetParameters() const
	{

		return fUserParameters;
	}

	GBool_t isCached() const {
		return fCached;
	}

	void SetCached(GBool_t cached) {
		fCached = cached;
	}

	const IteratorCache& GetCacheBegin() const {
		return fCacheBegin;
	}

	const IteratorCache& GetCacheEnd() const {
		return fCacheBegin + fNEvents;
	}

	const IteratorData& GetDataBegin() const {
		return fDataBegin;
	}

	const IteratorData& GetDataEnd() const {
		return fDataEnd;
	}

	GReal_t GetErrorDef() const {
		return fErrorDef;
	}

	void SetErrorDef(GReal_t errorDef) {
		fErrorDef = errorDef;
	}

	GBool_t isWeighted() const {
		return fWeighted;
	}

	void SetWeighted(GBool_t weighted) {
		fWeighted = weighted;
	}

	GLong_t GetNEvents() const {
		return fNEvents;
	}

	GReal_t GetSumW() const {
		return fSumW;
	}

	void SetSumW(GReal_t sumW) {
		fSumW = sumW;
	}

	GReal_t GetSumW2() const {
		return fSumW2;
	}

	void SetSumW2(GReal_t sumW2) {
		fSumW2 = sumW2;
	}

	std::unordered_map<size_t, GReal_t>& GetFcnCache() const {
		return fFCNCache;
	}

	PDF& GetPDF() const {
			return fPDF;
	}

	PDF& GetPDF()  {
				return fPDF;
		}



protected:

	GReal_t GetFCNValue(const std::vector<double>& parameters) const {

		size_t key = hydra::detail::hash_range(parameters.begin(),
				parameters.end());

		auto search = fFCNCache.find(key);
		GReal_t value = 0.0;

		//bool fnd=0;

		if (search != fFCNCache.end() && fFCNCache.size()>0) {
			value = search->second;
			//std::cout << "found NLL!" << std::endl;
			//fnd=1;
		} else {

			value = EvalFCN(parameters);

			fFCNCache[key] = value;

		}

		//cout << "Found: " << fnd << " Key: " << key << " Value: "<< std::setprecision(16) <<value << endl;
		return value;
	}

	GReal_t EvalFCN(const std::vector<double>& parameters) const {
		return static_cast<const estimator_type*>(this)->Eval(parameters);
	}

	void LoadFCNParameters()
	{
		std::cout << " LoadFCNParameters() in" << std::endl;
		std::vector<hydra::Parameter*> temp;


		fPDF.AddUserParameters(temp );

		std::cout << "<----------------" << std::endl;
		fUserParameters.SetVariables( temp);
		std::cout << " LoadFCNParameters() exit" << std::endl;

	}

	PDF& fPDF;
	IteratorData fDataBegin;
	IteratorData fDataEnd;
	IteratorCache fCacheBegin;
	GReal_t fSumW;
	GReal_t fSumW2;
	GReal_t fErrorDef;
	GLong_t fNEvents;
	GBool_t fWeighted;
	GBool_t fCached;
	mutable std::unordered_map<size_t, GReal_t> fFCNCache;
	hydra::UserParameters fUserParameters ;
};

}  // namespace experimental

} //namespace hydra
#endif /* FCN_H_ */
