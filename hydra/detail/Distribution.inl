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
 * Distribution.inl
 *
 *  Created on: Feb 19, 2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DISTRIBUTION_INL_
#define DISTRIBUTION_INL_

namespace hydra {

/*
 * auto dist = Distribution<Functor>(FunctorObj);
 * auto x = dist(RngEngine);
 *
 */

template<typename Functor>
struct  Distribution: protected GenerationFormula<Functor>
{
  typedef GenerationFormula<Functor> formula_type;
  typedef typename formula_type::return_type  return_type;

protected:

  typedef hydra_thrust::uniform_real_distribution<return_type> uniform_type;

public:

 template<typename Engine>
 __hydra_host__ __hydra_device__
 return_type operator()(Engine& rng)
 {
	 return static_cast<const formula_type& >(*this).Generate(rng);
 }

 template<typename Engine>
 __hydra_host__ __hydra_device__
 return_type uniform(Engine& rng)
 {
	 return uniform_type(rng, {0.0, 1.0});
 }

};

}  // namespace hydra



#endif /* DISTRIBUTION_INL_ */
