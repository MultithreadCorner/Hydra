/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2017 Antonio Augusto Alves Junior
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
 * CompositeBase.h
 *
 *  Created on: 11/07/2016
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef COMPOSITE_BASE_H_
#define COMPOSITE_BASE_H_



#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/TypeTraits.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/base_functor.h>
#include <hydra/detail/Constant.h>
#include <hydra/Parameter.h>

namespace hydra {


template<typename F0, typename F1, typename... Fs >
struct  CompositeBase
{
	    //tag
	    typedef void hydra_functor_tag;

	    //typedefs
	    typedef typename detail::common_type_t<F0::return_type, F1::return_type, Fs::return_type...> return_type ;
		typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<typename F1::return_type, typename Fs::return_type...> argument_type;
		typedef typename HYDRA_EXTERNAL_NS::thrust::tuple<F1, Fs...> functors_type;


		CompositeBase()=delete;


		CompositeBase(F0 const& f0, F1 const& f1,  Fs const& ...fs):
		fIndex(-1),
		fCached(0),
		fFtorTuple(HYDRA_EXTERNAL_NS::thrust::make_tuple(f0, f1, fs...))
	  	{ }

		__host__ __device__ inline
		CompositeBase(CompositeBase<F0,F1,Fs...> const& other):
			fFtorTuple( other.GetFunctors() ),
			fIndex( other.GetIndex() ),
			fCached( other.IsCached() )
		{ }

		__host__ __device__ inline
		CompositeBase<F0,F1,Fs...>& operator=(CompositeBase<F0,F1,Fs...> const& other)
		{
			this->fFtorTuple = other.GetFunctors() ;
			this->fIndex = other.GetIndex() ;
			this->fCached = other.IsCached() ;
			return *this;
		}

		inline void AddUserParameters(std::vector<hydra::Parameter*>& user_parameters )
		{
			detail::add_parameters_in_tuple(user_parameters, fFtorTuple );
		}


		inline void SetParameters(const std::vector<double>& parameters){

			detail::set_functors_in_tuple(fFtorTuple, parameters);
		}

		inline size_t  GetParametersKey(){

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			std::vector<double> _temp(_parameters.size());

			for(size_t i=0; i< _parameters.size(); i++)
				_temp[i]= *(_parameters[i]);

			size_t key = detail::hash_range(_temp.begin(), _temp.end() );

			return key;
		}

		inline size_t GetNumberOfParameters() const {

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			return _parameters.size();

		}

		template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
		inline const hydra::Parameter& GetParameter(Int i) const {

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);


			return *(_parameters[i]);
		}

		inline const hydra::Parameter& GetParameter(const char* name) const {

			size_t i=0;

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			for(i=0; i<N; i++)
				if (strcmp(_parameters[i]->GetName(),name)==0) break;

			return *(_parameters[i]) ;
		}

		template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
		inline hydra::Parameter& Parameter(Int i) {

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			return *(_parameters[i]);
		}

		inline hydra::Parameter& Parameter(const char* name) {

			size_t i=0;

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			for(i=0; i<N; i++)
				if (strcmp(_parameters[i]->GetName(),name)==0) break;

			return *(_parameters[i]) ;
		}


		template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
		__host__ __device__ inline
		void SetParameter(Int i, hydra::Parameter const& value) {

			std::vector<hydra::Parameter*> _parameters;
			detail::set_functors_in_tuple(fFtorTuple, _parameters);

			(*_parameters[i])=value;
		}

		template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
		__host__ __device__ inline
		void SetParameter(Int i, double value) {
			fParameters[i]=value;
		}

		__host__ inline
		void SetParameter(const char* name, hydra::Parameter const& value) {

			size_t i=0;

			for(i=0; i<N; i++)
				if (strcmp(fParameters[i].GetName(),name)==0){
					fParameters[i]=value;
					break;
				}
		}

		__host__ inline
		void SetParameter(const char* name, double value) {

			size_t i=0;

			for(i=0; i<N; i++)
				if (strcmp(fParameters[i].GetName(),name)==0){
					fParameters[i]=value;
					break;
				}
		}

		__host__ __device__ inline
		functors_type GetFunctors() const {return this->fFtorTuple;}

		__host__ __device__ inline
		int GetIndex() const { return this->fIndex; }

		__host__ __device__ inline
		void SetIndex(int index) {this->fIndex = index;}

		__host__ __device__ inline
		bool IsCached() const
		{ return this->fCached;}

		__host__ __device__ inline
		void SetCached(bool cached=true)
		{ this->fCached = cached; }


	private:

	  	functors_type fFtorTuple;
		int  fIndex;
		bool fCached;

};





}

#endif /* COMPOSE_H_ */
