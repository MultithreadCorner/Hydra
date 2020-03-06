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
 * SimultaneousFCN.inl
 *
 *  Created on: 04/02/2020
 *      Author: Davide Brundu
 */

#ifndef SIMULTANEOUSFCN_INL_
#define SIMULTANEOUSFCN_INL_



namespace hydra {


template<typename ...ESTIMATORS>
class SimultaneousFCN< hydra_thrust::tuple<ESTIMATORS...> > final : public ROOT::Minuit2::FCNBase {

public:

	 typedef hydra_thrust::tuple<ESTIMATORS...> estimators_type;
   enum {nestimators=hydra_thrust::tuple_size<hydra_thrust::tuple<ESTIMATORS...>>::value};


	 /**
 	 * @brief SimultaneousFCN constructor
 	 *
 	 * @param estimators hydra FCN instances: hydra::LogLikelihoodFCN or hydra::SimultaneousFCN
 	 */
	SimultaneousFCN( ESTIMATORS&... estimators):
	   fEstimators( hydra_thrust::make_tuple(estimators...))
		 {
			 std::vector<Parameter*> _tmp_pars;
		  	std::vector<double> _tmp_errordefs;
			 LoadFCNParameters<0>(_tmp_pars);
			 LoadMinErrorDef<0>(_tmp_errordefs);
		 }


	SimultaneousFCN(SimultaneousFCN<hydra_thrust::tuple<ESTIMATORS...> > const& other):
		fEstimators(other.GetEstimators()), fErrorDef(other.GetErrorDef())
		{
			std::vector<Parameter*> _tmp_pars;
			LoadFCNParameters<0>(_tmp_pars);
		}


	SimultaneousFCN<hydra_thrust::tuple<ESTIMATORS...> >&
	operator=(SimultaneousFCN<hydra_thrust::tuple<ESTIMATORS...> > const& other)
		{
				if(this==&other) return  *this;
				SimultaneousFCN<hydra_thrust::tuple<ESTIMATORS...> >::operator=(other);
				return  *this;
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


		estimators_type& Estimators() {
				return fEstimators;
		}


		const estimators_type& GetEstimators() const {
		 		return fEstimators;
		 }


		double Up() const {
			return fErrorDef;
		}


		virtual double operator()(std::vector<double> const& parameters) const {
			double result = 0.0;
			eval<0>(parameters, result);
			return result;
		}


private:

  double fErrorDef;
	estimators_type fEstimators;
	hydra::UserParameters fUserParameters ;


	template<size_t I>
	typename std::enable_if< (I==nestimators), void>::type
	LoadMinErrorDef(std::vector<double>& _tmp_errordefs){
		fErrorDef = *(std::min_element(_tmp_errordefs.begin() , _tmp_errordefs.end()));
	}

	template<size_t I=0>
	typename std::enable_if< (I<nestimators), void>::type
	LoadMinErrorDef(std::vector<double>& _tmp_errordefs){
		_tmp_errordefs.push_back(hydra_thrust::get<I>(fEstimators).GetErrorDef());
		LoadMinErrorDef<I+1>(_tmp_errordefs);
	}



	template<size_t I>
	typename std::enable_if< (I==nestimators), void>::type
	LoadFCNParameters(std::vector<Parameter*>& _tmp_pars){
		fUserParameters.SetVariables(_tmp_pars);
	}

	template<size_t I=0>
	typename std::enable_if< (I<nestimators), void>::type
	LoadFCNParameters(std::vector<Parameter*>& _tmp_pars){
		auto vars = hydra_thrust::get<I>(fEstimators).GetParameters().GetVariables();
		for (auto param_ptr : vars) _tmp_pars.push_back(param_ptr);
		LoadFCNParameters<I+1>(_tmp_pars);
	}



	template<size_t I>
	typename std::enable_if< (I==nestimators), void>::type
	eval( std::vector<double> const& , double&   ) const {}

	template<size_t I=0>
	typename std::enable_if< (I<nestimators), void>::type
	eval( std::vector<double> const& parameters, double& result ) const
	{
		result += hydra_thrust::get<I>(fEstimators)(parameters);
	  eval<I+1>(parameters, result );
	}

};



template<typename ...ESTIMATORS>
typename std::enable_if<hydra::detail::valid_estimators<ESTIMATORS...>::value, SimultaneousFCN< hydra_thrust::tuple<ESTIMATORS...>> >::type
make_simultaneous_fcn( ESTIMATORS& ... estimators)
{
  return SimultaneousFCN< hydra_thrust::tuple<ESTIMATORS...>>(estimators...);
}




}  // namespace hydra





#endif /* SIMULTANEOUSFCN_INL_ */
