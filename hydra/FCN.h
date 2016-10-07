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

#ifndef FCN_H_
#define FCN_H_

#include <hydra/detail/Config.h>

#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/Hash.h>
#include <hydra/detail/functors/LogLikelihood.h>

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

template<typename ESTIMATOR, typename IteratorData, typename IteratorCache>
class FCN: public ROOT::Minuit2::FCNBase {
public:

	FCN(IteratorData begin, IteratorData end) :
		fDataBegin(begin),
		fDataEnd(end),
		fCacheBegin(IteratorCache()),
		fNEvents(thrust::distance(begin, end)),
		fErrorDef(0.5),
		fWeighted(kFalse),
		fCached(kFalse),
		fFCNCache(std::unordered_map<size_t, GReal_t>())
{
		typename IteratorData::value_type init;

		auto point = thrust::reduce(begin, end, init);
		fSumW  = point.GetWeight() ;
		fSumW2 = point.GetWeight2();
}

	FCN(IteratorData begin, IteratorData end, IteratorCache cbegin) :
		fDataBegin(begin),
		fDataEnd(end),
		fCacheBegin(cbegin),
		fNEvents(thrust::distance(begin, end)),
		fErrorDef(0.5),
		fWeighted(kFalse),
		fCached(kTrue),
		fFCNCache(std::unordered_map<size_t, GReal_t>())
	{
		typename IteratorData::value_type init;

		auto point = thrust::reduce(begin, end, init);
		fSumW  = point.GetWeight() ;
		fSumW2 = point.GetWeight2();
	}

	FCN(FCN<ESTIMATOR, IteratorData, IteratorCache> const& other) :
		ROOT::Minuit2::FCNBase(other),
		fDataBegin(other.GetDataBegin()),
		fDataEnd(other.GetDataEnd()),
		fCacheBegin(other.GetCacheBegin()),
		fNEvents(other.GetNEvents()),
		fErrorDef(other.ErrorDef()),
		fWeighted(other.isWeighted()),
		fCached(other.isCached()),
		fFCNCache(other.GetFcnCache()),
		fSumW(other.GetSumW()),
		fSumW2(other.GetSumW2())
	{}

	FCN<ESTIMATOR, IteratorData, IteratorCache>&
	operator=(FCN<ESTIMATOR, IteratorData, IteratorCache> const& other)
	{
		ROOT::Minuit2::FCNBase::operator = (other);
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

private:

	GReal_t GetFCNValue(const std::vector<double>& parameters) const {

		size_t key = detail::hash_range(parameters.begin(),
				parameters.end());

		auto search = fFCNCache.find(key);
		GReal_t value = 0.0;

		if (search != fFCNCache.end() && fFCNCache.size()>0) {
			value = search->second;
		} else {
			//g_pages_mutex.lock();
			value = EvalFCN(parameters);

			fFCNCache[key] = value;
			//g_pages_mutex.unlock();
		}

		return value;
	}

	GReal_t EvalFCN(const std::vector<double>& parameters) const {
		return static_cast<const ESTIMATOR*>(this)->Eval(parameters);
	}

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

};

} //namespace hydra
#endif /* FCN_H_ */
