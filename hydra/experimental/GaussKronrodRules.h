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
 * GaussKronrodRule.h
 *
 *  Created on: Jan 25, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODRULES_H_
#define GAUSSKRONRODRULES_H_

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/detail/Print.h>
#include <hydra/experimental/GaussKronrodRule.h>
#include <initializer_list>


namespace hydra {

namespace experimental {


template<size_t N>
struct GaussKronrodRuleSelector;


template<>
struct GaussKronrodRuleSelector<15>
{
	GaussKronrodRuleSelector():
		fRule(
				{
				//Kronrod nodes
				0.000000000000000000000000000000000e+00,//*
				2.077849550078984676006894037732449e-01,//
				4.058451513773971669066064120769615e-01,//*
				5.860872354676911302941448382587296e-01,//
				7.415311855993944398638647732807884e-01,//*
				8.648644233597690727897127886409262e-01,//
				9.491079123427585245261896840478513e-01,//*
				9.914553711208126392068546975263285e-01 //
				},
				{
				//Gauss Weights
				4.179591836734693877551020408163265e-01,//*
				0.0000000000000000000000000000000000000,//
				3.818300505051189449503697754889751e-01,//*
				0.0000000000000000000000000000000000000,//
				2.797053914892766679014677714237796e-01,//*
				0.0000000000000000000000000000000000000,//
				1.294849661688696932706114326790820e-01,//*
				0.0000000000000000000000000000000000000 //
				},
				{
				//Kronrod Weights
				2.094821410847278280129991748917143e-01,
				2.044329400752988924141619992346491e-01,
				1.903505780647854099132564024210137e-01,
				1.690047266392679028265834265985503e-01,
				1.406532597155259187451895905102379e-01,
				1.047900103222501838398763225415180e-01,
				6.309209262997855329070066318920429e-02,
				2.293532201052922496373200805896959e-02 }
			)
		{}

GaussKronrodRule<15> fRule;

};



}//namespace experimental

}//namespace hydra


#endif /* GAUSSKRONRODRULE_H_ */
