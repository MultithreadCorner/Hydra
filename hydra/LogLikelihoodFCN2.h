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
 * LogLikelihoodFCN2.h
 *
 *  Created on: 13/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef LOGLIKELIHOODFCN2_H_
#define LOGLIKELIHOODFCN2_H_

#include <hydra/FCN2.h>

namespace hydra {

template<typename PDF, typename Iterator, typename... Extensions>
class LogLikelihoodFCN2;

/*
template<typename PDF, typename Iterator>
class LogLikelihoodFCN2<PDF, Iterator>; public FCN2<LogLikelihoodFCN2<PDF,Iterator > >;

template<typename PDF, typename Iterator, typename Weights>
class LogLikelihoodFCN2<PDF, Iterator, Weights>; public FCN2<LogLikelihoodFCN2<PDF,Iterator, Weights> >;

template<typename PDF, typename Iterator, typename Cache>
class LogLikelihoodFCN2<PDF, Iterator, Cache>; public FCN2<LogLikelihoodFCN2<PDF,Iterator, Cache> >;

template<typename PDF, typename Iterator, typename Weights, typename Cache>
class LogLikelihoodFCN2<PDF, Iterator, Weights,Cache>; public FCN2<LogLikelihoodFCN2<PDF,Iterator, Weights,Cache> >;
*/

}  // namespace hydra

#include<hydra/detail/LogLikelihoodFCN1.inl>

#endif /* LOGLIKELIHOODFCN2_H_ */
