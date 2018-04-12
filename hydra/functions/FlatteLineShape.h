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
 * FlatteLineShape.h
 *
 *  Created on: 03/09/2018 Based on Goofit Flatte Code
 *      Author: Juan B de S Leite
 *
 *
 */

#ifndef HYDRA_EXAMPLES_AND_DOCUMENTATION_FLATTELINESHAPE_H
#define HYDRA_EXAMPLES_AND_DOCUMENTATION_FLATTELINESHAPE_H

#include <hydra/detail/Config.h>
#include <hydra/Types.h>
#include <hydra/Function.h>
#include <hydra/detail/utility/CheckValue.h>
#include <hydra/Parameter.h>
#include <hydra/Tuple.h>
#include <hydra/Complex.h>
#include <hydra/functions/Utils.h>
#include <hydra/functions/BlattWeisskopfFunctions.h>
#include <hydra/Vector4R.h>
#include <tuple>
#include <limits>
#include <stdexcept>
#include <assert.h>
#include <utility>
#include <cmath>

#include <TMath.h>

namespace hydra {

    /*
     * @FlatteLineShape
     * The  Flatté  formulation  is  used  when  a second channel opens close to a resonance.
     * T(m_{ab}) = 1 / ( m_{r}^{2} - m_{ab}^{2} - i(#rho_{1}*g_{1}^{2} + #rho_{2}*g_{2}^{2})
     * where g_{1}^{2} + g_{2}^{2} = m_{0}*#Gamma_{r}
     * @tparam unsigned int CHANNEL
    */


    template<unsigned int CHANNEL, unsigned int ArgIndex = 0>
    class FlatteLineShape : public BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 1> {
        using BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 1>::_par;

    public:

        FlatteLineShape() = delete;

/**
 *
 * @param mass resonance mass.
 * @param rho1
 * @param rh2
 * @param daugther1_mass resonance daughter particle 1 mass
 * @param daugther2_mass resonance daughter particle 2 mass
 * @param daugther3_mass daughter particle 2 mass
 * @param radi decay vertex radio.
 */
        FlatteLineShape(hydra::Parameter const& mean,
                        std::array<std::array<double,3>,2> params) :
                BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>,1>{mean},
                fParams(params)
                {}

        __hydra_host__  __hydra_device__
        FlatteLineShape(FlatteLineShape<CHANNEL, ArgIndex> const &other) :
                BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 1>(other),
                fParams(other.GetParams())
                 {}

        __hydra_host__  __hydra_device__
        FlatteLineShape<CHANNEL, ArgIndex> &
        operator=(FlatteLineShape<CHANNEL, ArgIndex> const &other) {
            if (this == &other) return *this;

            BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>,
                    hydra::complex<double>, 1>::operator=(other);

            fParams = other.GetParams();

            return *this;
        }

        __hydra_host__  __hydra_device__ inline
        std::array<std::array<double,3>,2> GetParams() const {
            return fParams;
        }

        __hydra_host__  __hydra_device__ inline
        void SetParams(std::array<std::array<double,3>,2> _params) {
            fParams = _params;
        }

        template<typename T>
        __hydra_host__ __hydra_device__ inline
        hydra::complex<double> Evaluate(unsigned int, T *x) const {

            const double m = x[ArgIndex];
            const double resonance_mass = _par[0];

            return LineShape( m , resonance_mass) ;

        }

        template<typename T>
        __hydra_host__ __hydra_device__ inline
        hydra::complex<double> Evaluate(T x) const {

			const double m = get<ArgIndex>(x);
            const double resonance_mass = _par[0];

            return LineShape( m , resonance_mass) ;
        }

    private:

        __hydra_host__ __hydra_device__ inline
        hydra::complex<double>
        sqrtCplx(const double in) const { return (in > 0) ? hydra::complex<double>(::sqrt(in), 0.0) : hydra::complex<double>(0.0, ::sqrt(-in)); }


        __hydra_host__ __hydra_device__ inline hydra::complex<double> LineShape(const double s, const double resonance_mass) const  {

            hydra::complex<double> w;



            for(size_t i = 0; i < fParams.size() ; i++) {

                double m1a = fParams[i][0];
                double m1b = fParams[i][1];
                double g   = fParams[i][2];

                w += g * g * sqrtCplx((1 - (m1a - m1b) * (m1a - m1b) / s*s ) *
                                      (1 - (m1a + m1b) * (m1a + m1b) / s*s ));

            }

            hydra::complex<double> denom =   resonance_mass - s*s - hydra::complex<double>(0.0,1.0)*w;

            hydra::complex<double> ampl = hydra::complex<double>(1.0,0.0)/denom;

            return ampl;

        }

       std::array<std::array<double,3>,2> fParams;

    };

}
#endif //HYDRA_EXAMPLES_AND_DOCUMENTATION_FLATTELINESHAPE_H
