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
 * AddPdfBase.h
 *
 *  Created on: 09/10/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef ADDPDFBASE_H_
#define ADDPDFBASE_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/detail/utility/Generic.h>
#include <hydra/detail/FunctorTraits.h>
#include <utility>

namespace hydra {

namespace detail {


template<typename PDF1, typename PDF2, typename ...PDFs>
class AddPdfChecker:  public all_true<
detail::is_hydra_pdf<PDF1>::value,
detail::is_hydra_pdf<PDF2>::value,
detail::is_hydra_pdf<PDFs>::value...>{} ;

template<typename PDF1, typename PDF2, typename ...PDFs>
class AddPdfBase: public std::enable_if<AddPdfChecker<PDF1,PDF2,PDFs...>::value>
{};

}  // namespace detail

}  // namespace hydra


#endif /* ADDPDFBASE_H_ */
