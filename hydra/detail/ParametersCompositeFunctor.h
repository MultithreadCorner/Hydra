/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * ParametersCompositeFunctor.h
 *
 *  Created on: 08/09/2018
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef PARAMETERSCOMPOSITEFUNCTOR_H_
#define PARAMETERSCOMPOSITEFUNCTOR_H_




#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/detail/Constant.h>
#include <hydra/Parameter.h>
#include <hydra/Placeholders.h>

namespace hydra {

namespace detail {

template<typename F0, typename F1, typename... Fs >
class  ParametersCompositeFunctor
{

public:

	typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<F0, F1, Fs...> functors_type;

	ParametersCompositeFunctor()=delete;

	ParametersCompositeFunctor(F0 const& f0, F1 const& f1,  Fs const& ...fs):
		fFtorTuple(HYDRA_EXTERNAL_NS::thrust::make_tuple(f0, f1, fs...))
	{ }

	__hydra_host__ __hydra_device__
	inline ParametersCompositeFunctor(ParametersCompositeFunctor<F0,F1,Fs...> const& other):
	fFtorTuple( other.GetFunctors() )
	{}


	__hydra_host__ __hydra_device__
	inline ParametersCompositeFunctor<F0,F1,Fs...>& operator=(ParametersCompositeFunctor<F0,F1,Fs...> const& other) {
		if(this == &other) return *this;

		this->fFtorTuple = other.GetFunctors() ;

		return *this;
	}

	inline void PrintRegisteredParameters() {

		HYDRA_CALLER ;
		HYDRA_MSG << "Registered parameters begin:\n" << HYDRA_ENDL;
		detail::print_parameters_in_tuple(fFtorTuple);
		HYDRA_MSG <<"Registered parameters end.\n" << HYDRA_ENDL;
		return;
	}

	inline void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters ) {

		detail::add_parameters_in_tuple(user_parameters, fFtorTuple );
	}


	inline void SetParameters(const std::vector<double>& parameters){

		detail::set_functors_in_tuple(fFtorTuple, parameters);
		Update();
	}

	inline size_t  GetParametersKey(){

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		std::vector<double> _temp(_parameters.size());

		for(size_t i=0; i< _parameters.size(); i++)
			_temp[i]= *(_parameters[i]);

		size_t key = detail::hash_range(_temp.begin(), _temp.end() );

		return key;
	}

	inline size_t GetNumberOfParameters() const {

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		return _parameters.size();

	}

	template<typename Int,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline const hydra::Parameter& GetParameter(Int i) const {

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );


		return *(_parameters[i]);
	}

	inline const hydra::Parameter& GetParameter(const char* name) const {

		size_t i=0;

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		for(i=0; i< _parameters.size(); i++)
			if (strcmp(_parameters[i]->GetName(),name)==0) break;

		return *(_parameters[i]) ;
	}

	template<typename Int,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline hydra::Parameter& Parameter(Int i) {

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		return *(_parameters[i]);
	}

	inline hydra::Parameter& Parameter(const char* name) {

		size_t i=0;

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		for(i=0; i<_parameters.size(); i++)
			if (strcmp(_parameters[i]->GetName(),name)==0) break;

		return *(_parameters[i]) ;
	}


	template<typename Int,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline void SetParameter(Int i, hydra::Parameter const& value) {

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		*(_parameters[i])=value;
	}

	template<typename Int,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	inline void SetParameter(Int i, double value) {
		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		*(_parameters[i])=value;
	}

	inline	void SetParameter(const char* name, hydra::Parameter const& value) {

		size_t i=0;

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );

		for(i=0; i<_parameters.size(); i++)
			if (strcmp(_parameters[i]->GetName(),name)==0){
				*(_parameters[i])=value;
				break;
			}
	}

	inline	void SetParameter(const char* name, double value) {

		size_t i=0;

		std::vector<hydra::Parameter*> _parameters;
		detail::add_parameters_in_tuple(_parameters, fFtorTuple );


		for(i=0; i<_parameters.size(); i++)
			if (strcmp(_parameters[i]->GetName(),name)==0){
				*(_parameters[i])=value;
				break;
			}
	}

	__hydra_host__ __hydra_device__ inline
	const functors_type& GetFunctors() const {return fFtorTuple;}

	template<unsigned int I>
	inline typename HYDRA_EXTERNAL_NS::thrust::tuple_element<I,functors_type>::type&
	GetFunctor(hydra::placeholders::placeholder<I> const& ){

		return HYDRA_EXTERNAL_NS::thrust::get<I>(fFtorTuple);
	}

	virtual void Update(){};


	virtual ~ParametersCompositeFunctor()=default;

private:

	functors_type fFtorTuple;

};

}  // namespace detail

} // namespace hydra


#endif /* PARAMETERSCOMPOSITEFUNCTOR_H_ */
