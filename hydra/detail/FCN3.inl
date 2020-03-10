/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2020 Antonio Augusto Alves Junior
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
 * FCN3.inl
 *
 *  Created on: 07/03/2020
 *      Authors: Davide Brundu and Antonio Augusto Alves Junior
 *
 *  Note: Davide contributed the initial version of
 *  this code in the PR#83 ( https://github.com/MultithreadCorner/Hydra/pull/83)
 *  I (A.A.A.Jr.) renamed some files, added features, corrections and moved some code around.
 */

#ifndef FCN3_INL_
#define FCN3_INL_

#include <algorithm>
#include <future>
#include <vector>

#include <hydra/detail/EstimatorTraits.h>

namespace hydra {


template<typename ...ESTIMATORS>
class FCN< hydra_thrust::tuple<FCN<ESTIMATORS>...>, false >:public ROOT::Minuit2::FCNBase
{

	typedef hydra_thrust::tuple< FCN<ESTIMATORS>...> estimator_type;

public:

	enum { nfcns = hydra_thrust::tuple_size<hydra_thrust::tuple<FCN<ESTIMATORS>...>>::value};


	FCN( FCN<ESTIMATORS>const&... fcns):
		fErrorDef(0.5),
		fFCNS( hydra_thrust::make_tuple( fcns...))
	{
		std::initializer_list<double> error_defs{fcns.GetErrorDef()...};

		fErrorDef = *std::min_element( error_defs.begin(),  error_defs.end());

		LoadFCNParameters();
	}


	FCN(FCN<estimator_type, false> const& other):
		ROOT::Minuit2::FCNBase(other),
		fFCNS(other.GetFCNS()),
		fErrorDef(other.GetErrorDef())
	{
		LoadFCNParameters();
	}


	FCN<hydra_thrust::tuple<FCN<ESTIMATORS>...>, false >&
	operator=(FCN<hydra_thrust::tuple<FCN<ESTIMATORS>...>, false > const& other)
	{
		if(this==&other) return *this;

		ROOT::Minuit2::FCNBase::operator=(other);
		fFCNS=other.GetFCNS();
		fErrorDef=other.GetErrorDef();
		LoadFCNParameters();

		return  *this;
	}

	hydra::UserParameters const& GetParameters() const {
		return fUserParameters;
	}

	hydra::UserParameters& GetParameters() {
		return fUserParameters;
	}

	double GetErrorDef() const {
		return fErrorDef;
	}

	void  SetErrorDef(double error){
		fErrorDef=error;
	}

	estimator_type& FCNS() {
		return fFCNS;
	}

	const estimator_type& GetFCNS() const {
		return fFCNS;
	}

	double Up() const {
		return fErrorDef;
	}

	virtual double operator()(std::vector<double> const& parameters) const {

		return InvokeFCNS(parameters);
	}

private:

	template<size_t I>
	typename std::enable_if<(I==nfcns), void>::type
	load_fcn_parameters_helper(std::vector<Parameter*>&){}

	template<size_t I=0>
	typename std::enable_if< (I<nfcns), void>::type
	load_fcn_parameters_helper( std::vector<Parameter*>& pars){

		auto vars = hydra_thrust::get<I>(fFCNS).GetParameters().GetVariables();

		for (auto param : vars)
			pars.push_back(param);

		load_fcn_parameters_helper<I+1>(pars);
	}

	void LoadFCNParameters() {

		std::vector<Parameter*> pars{};

		load_fcn_parameters_helper(pars);

		fUserParameters.SetVariables(pars);
	}

	template<size_t I>
	typename std::enable_if< (I==nfcns), void>::type
	add_tasks( std::vector<double> const& parameters, std::vector<std::future<double>>&  ) const {}

	template<size_t I=0>
	typename std::enable_if< (I<nfcns), void>::type
	add_tasks(std::vector<double> const& parameters, std::vector<std::future<double>>& tasks ) const
	{
		tasks.push_back(std::async(
				std::launch::async,
				[this](std::vector<double> const& p) {
		        	return hydra_thrust::get<I>(fFCNS)(p);
		        },
				parameters ) );

		add_tasks<I+1>(parameters, tasks );
	}

	double InvokeFCNS(std::vector<double> const& parameters) const
	{
		 std::vector<std::future<double>> tasks;
		 add_tasks( parameters,  tasks);
		 double result = 0;
		 for(auto& partial_result: tasks){
			 partial_result.wait();
			 result += partial_result.get();
		 }
		 return result;

	}

	double fErrorDef;
	estimator_type fFCNS;
	UserParameters fUserParameters ;

};
template<typename ...ESTIMATORS>
FCN< hydra_thrust::tuple< FCN<ESTIMATORS>...>, false >
make_simultaneous_fcn( FCN<ESTIMATORS>const& ... fcns)
{
 return	FCN< hydra_thrust::tuple< FCN<ESTIMATORS>...> , false>(fcns...);
}

}  // namespace hydra


#endif /* FCN3_INL_ */
