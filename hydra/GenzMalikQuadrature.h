/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2018 Antonio Augusto Alves Junior
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
 * GenzMalikQuadrature.h
 *
 *  Created on: Feb 28, 2017
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */

#ifndef GENZMALIKQUADRATURE_H_
#define GENZMALIKQUADRATURE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/GenzMalikRule.h>
#include <hydra/detail/GenzMalikBox.h>
#include <hydra/multivector.h>
#include <hydra/detail/Integrator.h>
#include <hydra/detail/utility/Generic.h>

#include <algorithm>
#include <cmath>

namespace hydra {

template<  size_t N, typename  BACKEND>
class  GenzMalikQuadrature;

/**
 * \brief Non-adaptive Genz-Malik multidimensional quadrature
 *
 * Non-adaptive Genz-Malik multidimensional quadrature. This algorithm follows the original paper prescription
 * but does not implement the subdivision strategy yet.
 * A. C. Genz and A. A. Malik, "An adaptive algorithm for numeric integration over an N-dimensional rectangular region," J. Comput. Appl. Math. 6 (4), 295–302 (1980).
 * J. Berntsen, T. O. Espelid, and A. Genz, "An adaptive algorithm for the approximate calculation of multiple integrals," ACM Trans. Math. Soft. 17 (4), 437–451 (1991)
 */
template<  size_t N, hydra::detail::Backend  BACKEND>
class  GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND> >:
public Integrator<typename std::enable_if< (N>1),GenzMalikQuadrature<N, hydra::detail::BackendPolicy<BACKEND>>>::type  >
{

	typedef std::vector<detail::GenzMalikBox<N>> box_list_type;
	typedef typename box_list_type::iterator box_iterator;
	typedef typename box_list_type::const_iterator const_box_iterator;


	typedef typename GenzMalikRule<N, hydra::detail::BackendPolicy<BACKEND>>::abscissa_iterator rule_iterator;
	typedef typename GenzMalikRule<N, hydra::detail::BackendPolicy<BACKEND>>::const_abscissa_iterator const_rule_iterator;

public:

	GenzMalikQuadrature()=delete;

	/**
	 * Genz-Malik multidimensional quadrature constructor.
	 * @param LowerLimit : std::array with the lower limits of integration
	 * @param UpperLimit : std::array with the upper limits of integration
	 * @param grid       : std::array with the number of divisions per dimension
	 */
	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			std::array<size_t, N> const& grid);


	/**
	 * Genz-Malik multidimensional quadrature constructor.
	 * @param LowerLimit : std::array with the lower limits of integration
	 * @param UpperLimit : std::array with the upper limits of integration
	 * @param nboxes     : max number of multidimensional boxes
	 */
	GenzMalikQuadrature(std::array<GReal_t,N> const& LowerLimit,
			std::array<GReal_t,N> const& UpperLimit,
			size_t nboxes=10);

	/**
	 * Copy constructor.
	 * @param other : object on same back-end
	 */
	GenzMalikQuadrature( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>> const& other);

	/**
	 * Copy constructor.
	 * @param other : object on different back-end
	 */
	template< hydra::detail::Backend  BACKEND2>
	GenzMalikQuadrature( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND2>> const& other);

	/**
	 * Assignment operator
	 * @param other : object on same back-end
	 * @return
	 */
	GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>> const& other);

	/**
	 * Assignment operator
	 * @param other: object on different back-end
	 * @return
	 */
	template<hydra::detail::Backend BACKEND2>
	GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND>>&
	operator=( GenzMalikQuadrature<N,hydra::detail::BackendPolicy<BACKEND2>> const& other);

	/**
	 * Integrate method
	 * @param functor : integrand
	 * @return
	 */
	template<typename FUNCTOR>
	std::pair<GReal_t, GReal_t> Integrate(FUNCTOR const& functor);


	/**
	 * Print
	 */
	void Print()
	{
		fGenzMalikRule.Print();
		for(auto box: fBoxList )	box.Print();

	}

	const box_list_type& GetBoxList() const {
		return fBoxList;
	}

	const GenzMalikRule<N, hydra::detail::BackendPolicy<BACKEND>>& GetGenzMalikRule() const {
		return fGenzMalikRule;
	}

	void SetBoxList(const box_list_type& boxList) {
		fBoxList = boxList;
	}

	void SetGenzMalikRule(const GenzMalikRule<N, hydra::detail::BackendPolicy<BACKEND>>& genzMalikRule) {
		fGenzMalikRule = genzMalikRule;
	}


private:

	void GetGrid( size_t nboxes , std::array<size_t, N>& grid )
	{
		size_t ndivsion = std::llround( std::pow( 2.0, std::log2(double(nboxes))/double(N) ) );

		std::fill_n( grid.begin(), N,  ndivsion );

	}


	GenzMalikRule<  N,  hydra::detail::BackendPolicy<BACKEND>> fGenzMalikRule;
	box_list_type fBoxList;

};



} // namespace hydra

#include <hydra/detail/GenzMalikQuadrature.inl>

#endif /* GENZMALIKQUADRATURE_H_ */
