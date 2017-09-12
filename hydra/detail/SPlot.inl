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
 * SPlot.inl
 *
 *  Created on: 12/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPLOT_INL_
#define SPLOT_INL_


namespace hydra {

template <typename PDF1, typename PDF2, typename ...PDFs>
template<typename InputIterator, typename OutputIterator>
inline void SPlot<PDF1,PDF2,PDFs...>::Generate(InputIterator in_begin, InputIterator in_end,
			OutputIterator out_begin) const
			{

}


} // namespace hydra

#endif /* SPLOT_INL_ */
