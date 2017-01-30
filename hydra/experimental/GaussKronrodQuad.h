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
 * GaussKronrodQuad.h
 *
 *  Created on: Jan 25, 2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef GAUSSKRONRODQUAD_H_
#define GAUSSKRONRODQUAD_H_



namespace hydra {

namespace experimental {


template<size_t N>
struct GaussKronrodRules
{
	GReal_t X[N];
	GReal_t GaussWeight[(N -1)/2];
	GReal_t KronrodWeight[N];

};

/*
 template<size_t GaussN, size_t KronrodN>
struct GaussKronrodRules
{
	double X[ (KronrodN-1)/2 + 1 ];
	double GaussWeight[(GaussN-1)/2 +1 ];
    double KronrodWeight[(KronrodN-1)/2 + 1 ];

};

template<size_t GaussN, size_t KronrodN>
void print_rule( GaussKronrodRules<GaussN, KronrodN> const& rule )
{

    for(size_t i=0; i<(KronrodN-1)/2+1; i++ )
    {
        std::cout << "Gauss-Konrod x[" << i << "] = " << rule.X[i] << std::endl;
        std::cout << "Gauss-Konrod w[" << i << "] = " << rule.KronrodWeight[i] << std::endl;
    }

    for(size_t i=0; i<(GaussN-1)/2+1; i++ )
    {
        std::cout << "Gauss x[" << i << "] = " << rule.X[2*i] << std::endl;
        std::cout << "Gauss w[" << i << "] = " << rule.GaussWeight[i] << std::endl;
    }


}

*/

}

}


#endif /* GAUSSKRONRODQUAD_H_ */
