/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * GenzMalikRule.h
 *
 *  Created on: 22/02/2017
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef GENZMALIKRULE_H_
#define GENZMALIKRULE_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/multiarray.h>
//thrust
#include <hydra/detail/external/thrust/tuple.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/iterator/detail/tuple_of_iterator_references.h>
//std
#include <type_traits>
#include <array>
#include <initializer_list>
#include <math.h>
#include <vector>
#include <tuple>
#include <utility>
#include <algorithm>

namespace hydra {


template<typename T>
class GenzMalikRuleBase{};

template<size_t DIM, typename BACKEND>
class GenzMalikRule;

/**
 * \ingroup numerical_integration
 *
 * \brief Class representing Genz-Malik rule.
 *
 */
template<size_t DIM, hydra::detail::Backend   BACKEND>
class GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND>>:
GenzMalikRuleBase<typename std::enable_if< (DIM>1), void >::type >
{
	typedef  hydra::detail::BackendPolicy<BACKEND> system_type;

public:

	//container for abscissas {x_1, x_2, ... , x_n} weights
	typedef multiarray<double, DIM+4, system_type > storage_type;
	//iterators
	typedef typename storage_type::iterator iterator;
	typedef typename storage_type::const_iterator const_iterator;

	typedef typename storage_type::value_type abscissa_type;

	GenzMalikRule():
		fLambda2( sqrt(9.0/70.0) ),
		fLambda3( sqrt(9.0/10.0) ),
		fLambda4( sqrt(9.0/10.0) ),
		fLambda5( sqrt(9.0/19.0) ),
		fRule7Weight1(::pow(2.0,DIM)*(12824.0 - 9120.0*DIM + 400.0*DIM*DIM)/19683.0),
		fRule7Weight2(::pow(2.0,DIM)*(980.0/6561.0)),
		fRule7Weight3(::pow(2.0,DIM)*(1820.0 - 400.0*DIM)/19683.0),
		fRule7Weight4(::pow(2.0,DIM)*(200.0/19683.0)),
		fRule7Weight5(6859.0/19683.0),
		fRule5Weight1(::pow(2.0,DIM)*(729.0 - 950.0*DIM + 50.0*DIM*DIM)/729.0),
		fRule5Weight2(::pow(2.0,DIM)*(245.0/486.0)),
		fRule5Weight3(::pow(2.0,DIM)*(265.0 - 100.0*DIM)/1458.0),
		fRule5Weight4(::pow(2.0,DIM)*(25.0/729.0))
	{ SetAbscissas();}


	GenzMalikRule( GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND>> const& other):
		fLambda2( other.GetLambda2() ),
		fLambda3( other.GetLambda3() ),
		fLambda4( other.GetLambda4() ),
		fLambda5( other.GetLambda5() ),
		fRule7Weight1( other.GetRule7Weight1() ),
		fRule7Weight2( other.GetRule7Weight2() ),
		fRule7Weight3( other.GetRule7Weight3() ),
		fRule7Weight4( other.GetRule7Weight4() ),
		fRule7Weight5( other.GetRule7Weight5() ),
		fRule5Weight1( other.GetRule5Weight1() ),
		fRule5Weight2( other.GetRule5Weight2() ),
		fRule5Weight3( other.GetRule5Weight3() ),
		fRule5Weight4( other.GetRule5Weight4() ),
		fAbscissas( other.GetAbscissas() )
	{}

	template<hydra::detail::Backend BACKEND2>
	GenzMalikRule( GenzMalikRule<DIM, hydra::detail::BackendPolicy<BACKEND2>> const& other):
	fLambda2( other.GetLambda2() ),
	fLambda3( other.GetLambda3() ),
	fLambda4( other.GetLambda4() ),
	fLambda5( other.GetLambda5() ),
	fRule7Weight1( other.GetRule7Weight1() ),
	fRule7Weight2( other.GetRule7Weight2() ),
	fRule7Weight3( other.GetRule7Weight3() ),
	fRule7Weight4( other.GetRule7Weight4() ),
	fRule7Weight5( other.GetRule7Weight5() ),
	fRule5Weight1( other.GetRule5Weight1() ),
	fRule5Weight2( other.GetRule5Weight2() ),
	fRule5Weight3( other.GetRule5Weight3() ),
	fRule5Weight4( other.GetRule5Weight4() ),
	fAbscissas( other.GetAbscissas() )
	{}

	GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>>&
	operator=( GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>> const& other)
	{

		if(this==&other) return *this;

		fLambda2 = other.GetLambda2() ;
		fLambda3 = other.GetLambda3() ;
		fLambda4 = other.GetLambda4() ;
		fLambda5 = other.GetLambda5() ;
		fRule7Weight1 = other.GetRule7Weight1() ;
		fRule7Weight2 = other.GetRule7Weight2() ;
		fRule7Weight3 = other.GetRule7Weight3() ;
		fRule7Weight4 = other.GetRule7Weight4() ;
		fRule7Weight5 = other.GetRule7Weight5() ;
		fRule5Weight1 = other.GetRule5Weight1() ;
		fRule5Weight2 = other.GetRule5Weight2() ;
		fRule5Weight3 = other.GetRule5Weight3() ;
		fRule5Weight4 = other.GetRule5Weight4() ;
		fAbscissas = other.GetAbscissas() ;

		return *this;
	}

	template<hydra::detail::Backend BACKEND2>
	inline GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND>>&
	operator=( GenzMalikRule<DIM,hydra::detail::BackendPolicy<BACKEND2>> const& other)
	{

		if(this==&other) return *this;

		fLambda2 = other.GetLambda2() ;
		fLambda3 = other.GetLambda3() ;
		fLambda4 = other.GetLambda4() ;
		fLambda5 = other.GetLambda5() ;
		fRule7Weight1 = other.GetRule7Weight1() ;
		fRule7Weight2 = other.GetRule7Weight2() ;
		fRule7Weight3 = other.GetRule7Weight3() ;
		fRule7Weight4 = other.GetRule7Weight4() ;
		fRule7Weight5 = other.GetRule7Weight5() ;
		fRule5Weight1 = other.GetRule5Weight1() ;
		fRule5Weight2 = other.GetRule5Weight2() ;
		fRule5Weight3 = other.GetRule5Weight3() ;
		fRule5Weight4 = other.GetRule5Weight4() ;
		fAbscissas = other.GetAbscissas() ;

		return *this;
	}

	void Print()
	{
		HYDRA_MSG << HYDRA_ENDL;
		HYDRA_MSG << "Genz-Malik Rule begin:"              << HYDRA_ENDL;
		HYDRA_MSG << "(weight #5, weight #7, ...{abscissas})" << HYDRA_ENDL;
		for(auto row:fAbscissas)
		{
			HYDRA_MSG << row << HYDRA_ENDL;
		}
		HYDRA_MSG << "Number of function calls: "<< fAbscissas.size() << HYDRA_ENDL;
		HYDRA_MSG << "Genz-Malik Rule end."              << HYDRA_ENDL;
	}

	inline const storage_type& GetAbscissas() const
	{
		return fAbscissas;
	}

	inline void SetAbscissas(const storage_type& abscissas)
	{
		fAbscissas = abscissas;
	}

	inline iterator begin() {
		return fAbscissas.begin();
	}

	inline iterator end() {
		return fAbscissas.end();
	}


	inline const_iterator begin() const {
		return fAbscissas.begin();
	}

	inline const_iterator end()  const {
		return fAbscissas.end();
	}



	inline GReal_t GetLambda2() const
	{
		return fLambda2;
	}

	inline void SetLambda2(GReal_t lambda2)
	{
		fLambda2 = lambda2;
	}

	inline GReal_t GetLambda3() const
	{
		return fLambda3;
	}

	inline void SetLambda3(GReal_t lambda3)
	{
		fLambda3 = lambda3;
	}

	inline GReal_t GetLambda4() const
	{
		return fLambda4;
	}

	inline void SetLambda4(GReal_t lambda4)
	{
		fLambda4 = lambda4;
	}

	inline GReal_t GetLambda5() const
	{
		return fLambda5;
	}

	inline void SetLambda5(GReal_t lambda5)
	{
		fLambda5 = lambda5;
	}

	inline GReal_t GetRule5Weight1() const
	{
		return fRule5Weight1;
	}

	inline void SetRule5Weight1(GReal_t rule5Weight1)
	{
		fRule5Weight1 = rule5Weight1;
	}

	inline GReal_t GetRule5Weight2() const
	{
		return fRule5Weight2;
	}

	inline void SetRule5Weight2(GReal_t rule5Weight2)
	{
		fRule5Weight2 = rule5Weight2;
	}

	inline GReal_t GetRule5Weight3() const
	{
		return fRule5Weight3;
	}

	inline void SetRule5Weight3(GReal_t rule5Weight3)
	{
		fRule5Weight3 = rule5Weight3;
	}

	inline GReal_t GetRule5Weight4() const
	{
		return fRule5Weight4;
	}

	inline void SetRule5Weight4(GReal_t rule5Weight4)
	{
		fRule5Weight4 = rule5Weight4;
	}

	inline GReal_t GetRule7Weight1() const
	{
		return fRule7Weight1;
	}

	inline void SetRule7Weight1(GReal_t rule7Weight1)
	{
		fRule7Weight1 = rule7Weight1;
	}

	inline GReal_t GetRule7Weight2() const
	{
		return fRule7Weight2;
	}

	inline void SetRule7Weight2(GReal_t rule7Weight2)
	{
		fRule7Weight2 = rule7Weight2;
	}

	inline GReal_t GetRule7Weight3() const
	{
		return fRule7Weight3;
	}

	inline void SetRule7Weight3(GReal_t rule7Weight3)
	{
		fRule7Weight3 = rule7Weight3;
	}

	inline GReal_t GetRule7Weight4() const
	{
		return fRule7Weight4;
	}

	inline void SetRule7Weight4(GReal_t rule7Weight4)
	{
		fRule7Weight4 = rule7Weight4;
	}

	inline GReal_t GetRule7Weight5() const
	{
		return fRule7Weight5;
	}

	inline void SetRule7Weight5(GReal_t rule7Weight5)
	{
		fRule7Weight5 = rule7Weight5;
	}

private:

	inline void AddW1(){

		abscissa_type x{};
		HYDRA_EXTERNAL_NS::thrust::get<0>(x)= fRule5Weight1;
		HYDRA_EXTERNAL_NS::thrust::get<1>(x)= fRule7Weight1;
		HYDRA_EXTERNAL_NS::thrust::get<2>(x)= 6.0;
		HYDRA_EXTERNAL_NS::thrust::get<3>(x)= DIM;

		fAbscissas.push_back(x);
	}

	inline void AddW2(){

		std::array<double, DIM> x_temp{};

		//adding permutations of {lambda_2, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[DIM-1] = fLambda2;
        int dim = DIM-1;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight2, fRule7Weight2,-4.0, dim),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);
			dim--;

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

		//adding permutations of {-lambda_2, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[0] = -fLambda2;
		dim = 0;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight2, fRule7Weight2, -4.0, dim),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);
			dim++;

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

	}

	inline void AddW3(){

		std::array<double, DIM> x_temp{};

		//adding permutations of {lambda_3, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[DIM-1] = fLambda3;
		int dim = DIM-1;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight3,fRule7Weight3, 1.0, dim),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);
			dim--;

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

		//adding permutations of {-lambda_3, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[0] = -fLambda3;
		dim = 0;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight3,fRule7Weight3, 1.0, dim),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);
			dim++;

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

	}

	inline void AddW4(){

		std::array<double, DIM> x_temp{};

		//adding permutations of {lambda_4, lambda_4, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[DIM-1] = fLambda4;
		x_temp[DIM-2] = fLambda4;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight4,fRule7Weight4, 0.0,-1.0),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

		//adding permutations of {lambda_4,-lambda_4, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[0] = -fLambda4;
		x_temp[DIM-1] =  fLambda4;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight4,fRule7Weight4, 0.0,-1.0),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

		//adding permutations of {lambda_4,-lambda_4, 0, 0,...,0}
		for(auto& x:x_temp ) x=0;
		x_temp[0] = -fLambda4;
		x_temp[1] = -fLambda4;

		do {

			auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fRule5Weight4,fRule7Weight4, 0.0,-1.0),
					hydra::detail::arrayToTuple(x_temp));

			fAbscissas.push_back(x);

		} while( std::next_permutation(x_temp.begin(), x_temp.end()) );

	}

	inline void AddW5(){

		//adding permutations of {lambda_5,lambda_5,...,lambda_5}
		std::array<double, DIM> x_temp{};

		for(size_t i=0;i<DIM+1;i++){

			for(size_t j=0;j<DIM;j++){

				x_temp[j]= j<i ? -fLambda5: fLambda5;
			}

			do {

				auto x = HYDRA_EXTERNAL_NS::thrust::tuple_cat(
						HYDRA_EXTERNAL_NS::thrust::make_tuple(0.0,fRule7Weight5, 0.0,-1.0),
						hydra::detail::arrayToTuple(x_temp));

				fAbscissas.push_back(x);

			} while( std::next_permutation(x_temp.begin(), x_temp.end()) );


		}

	}



	inline void SetAbscissas()
	{
		fAbscissas.clear();
		AddW1();
		AddW2();
		AddW3();
		AddW4();
		AddW5();

	}



	GReal_t fLambda2;
	GReal_t fLambda3;
	GReal_t fLambda4;
	GReal_t fLambda5;

	GReal_t fRule7Weight1;
	GReal_t fRule7Weight2;
	GReal_t fRule7Weight3;
	GReal_t fRule7Weight4;
	GReal_t fRule7Weight5;

	GReal_t fRule5Weight1;
	GReal_t fRule5Weight2;
	GReal_t fRule5Weight3;
	GReal_t fRule5Weight4;

	storage_type fAbscissas;

};

} // namespace hydra

#endif /* GENZMALIKRULE_H_ */
