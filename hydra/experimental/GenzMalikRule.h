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
 * GenzMalikRule.h
 *
 *  Created on: 22/02/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GENZMALIKRULE_H_
#define GENZMALIKRULE_H_


//hydra
#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/experimental/multivector.h>
//thrust
#include <thrust/tuple.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
//std
#include <type_traits>
#include <array>
#include <initializer_list>
#include <math.h>
#include <vector>
#include <tuple>
#include <utility>

namespace hydra {

namespace experimental {


template<typename T>
struct GenzMalikRuleBase{};

template<size_t DIM, unsigned int BACKEND=hydra::host>
struct GenzMalikRule: GenzMalikRuleBase<typename std::enable_if< (DIM>1), void >::type >
	{


	//abscissa<0> -> degree five  weight
	//abscissa<1> -> degree seven weight
	//abscissa<(Index >=2)> -> multidimensional abscissa values
	typedef typename hydra::detail::tuple_type<DIM+2,GReal_t >::type abscissa_t;

	//system selection
	typedef hydra::detail::BackendTraits<BACKEND> system_t;

	//container template vector<abscissa> on device or host memory
	typedef typename system_t::template container<abscissa_t> super_t;

	//container
	typedef multivector<super_t> vector_abscissa_t;
	typedef typename multivector<super_t>::iterator vector_abscissa_iterator;
	typedef typename multivector<super_t>::const_iterator vector_abscissa_const_iterator;


		GenzMalikRule():
			fLambda2( sqrt(9.0/70.0) ),
			fLambda3( sqrt(9.0/10.0) ),
			fLambda4( sqrt(9.0/10.0) ),
			fLambda5( sqrt(9.0/19.0) )
			{

			GULong64_t factor  = twoN<DIM>();
			fRule7Weight1= factor*GReal_t(12824.0 - 9120.0*DIM + 400.0*DIM*DIM)/19683.0;
			fRule7Weight2= factor*GReal_t(980.0/6561.0);
			fRule7Weight3= factor*GReal_t(1820.0 - 400.0*DIM)/19683.0;
			fRule7Weight4= factor*GReal_t(200.0/19683.0);
			fRule7Weight5= factor*GReal_t(6859.0/19683.0);

			fRule5Weight1= factor*GReal_t(729.0 - 950.0*DIM + 50.0*DIM*DIM)/729.0;
			fRule5Weight2= factor*GReal_t(245.0/486.0);
			fRule5Weight3= factor*GReal_t(265.0 - 100.0*DIM)/1458.0;
			fRule5Weight4= factor*GReal_t(25.0/729.0);



			}

	private:

		template<size_t N=0>
		constexpr typename std::enable_if< (N==0), GULong64_t>::type twoN()
		{
			return 1;
		}

		template<size_t N=0>
		constexpr typename std::enable_if< (N>0), GULong64_t>::type twoN()
		{
			return 2*twoN<N-1>();
		}

		template<typename T, size_t N=1>
		void permute_abscissae(GReal_t rule5_weight, GReal_t rule7_weight,
				std::array<T, N> const& seed, vector_abscissa_t& container)
		{

			auto abscissa_temp = seed;
			std::sort(abscissa_temp.begin(), abscissa_temp.end());
			do{

				auto abscissa_tuple =
					thrust::tuple_cat(thrust::make_tuple(rule5_weight,rule7_weight),
						hydra::detail::arrayToTuple(abscissa_temp));

				container.push_back(abscissa_tuple);

			}while(std::next_permutation(abscissa_temp.begin(), abscissa_temp.end()));

		}


		template<typename T, size_t N=1>
		void get_seed( unsigned int order)
		{
		  typedef std::array<T, N> X_t;

		  switch( order )
		  {
		  case 0:
		  {

			  typename vector_abscissa_t::value_type x();
			  thrust::get<0>(x)= fRule5Weight1;
			  thrust::get<1>(x)= fRule7Weight1;
			  fAbscissae.push_back(x);
			  break;
		  }
			  // order = 2
		  case 2:
		  {
			  auto x = X_t();
			  x[0]= T(2);
			  permute_abscissae(fRule5Weight2, fRule7Weight2, x,  fAbscissae);
			  x[0]= T(-2);
			  permute_abscissae(fRule5Weight2, fRule7Weight2, x,  fAbscissae);

			  break;
		  }
			  // order = 3
		  case 3:
		  {
			  auto  x = X_t();
			  x[0]= T(3);
			  permute_abscissae(fRule5Weight3, fRule7Weight3, x,  fAbscissae);
			  x[0]= T(-3);
			  permute_abscissae(fRule5Weight3, fRule7Weight3, x,  fAbscissae);

			  break;
		  }
			  // order = 4
		  case 4:
		  {
			  auto 	  x = X_t();
			  x[0]= T(4); x[1]= T(4);
			  permute_abscissae(fRule5Weight4, fRule7Weight4, x,  fAbscissae);

			  x[0]= T(-4); x[1]= T(-4);
			  permute_abscissae(fRule5Weight4, fRule7Weight4, x,  fAbscissae);

			  x[0]= T(4); x[1]= T(-4);
			  permute_abscissae(fRule5Weight4, fRule7Weight4, x,  fAbscissae);

			  break;
		  }
			  // order = 5
		  case 5:
		  {
			  auto 	  x = X_t();
			  for(size_t i=0;i<N+1;i++)
			  {
				  for(size_t j=0;j<N;j++)
				  {
					  x[j]= T(j<i?-5:5);
				  }
				  permute_abscissae(0, fRule7Weight5, x,  fAbscissae);
			  }


			  break;
		  }
		  default:
			  break;

		  }

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

		vector_abscissa_t fAbscissae;

	};

}  // namespace experimental

} // namespace hydra

#endif /* GENZMALIKRULE_H_ */
