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
    class FlatteLineShape : public BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 3> {
        using BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 3>::_par;

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
        FlatteLineShape(hydra::Parameter const& mean, hydra::Parameter const& rho1 , hydra::Parameter const& rho2,
                        double mother_mass,
                        double daugther1_mass, double daugther2_mass, double daugther3_mass,
                        double radi) :
                BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 3>{mean, rho1, rho2},
                fDaughter1Mass(daugther1_mass),
                fDaughter2Mass(daugther2_mass),
                fBachelorMass(daugther3_mass),
                fMotherMass(mother_mass),
                fRadi(radi) {}

        __hydra_host__  __hydra_device__
        FlatteLineShape(FlatteLineShape<CHANNEL, ArgIndex> const &other) :
                BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 3>(other),
                fDaughter1Mass(other.GetDaughter1Mass()),
                fDaughter2Mass(other.GetDaughter2Mass()),
                fBachelorMass(other.GetDaughter3Mass()),
                fMotherMass(other.GetMotherMass()),
                fRadi(other.GetRadi()) {}

        __hydra_host__  __hydra_device__
        FlatteLineShape<CHANNEL, ArgIndex> &
        operator=(FlatteLineShape<CHANNEL, ArgIndex> const &other) {
            if (this == &other) return *this;

            BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>,
                    hydra::complex<double>, 3>::operator=(other);

            fDaughter1Mass = other.GetDaughter1Mass();
            fDaughter2Mass = other.GetDaughter2Mass();
            fBachelorMass = other.GetDaughter3Mass();
            fMotherMass = other.GetMotherMass();
            fRadi = other.GetRadi();

            return *this;
        }

        __hydra_host__  __hydra_device__ inline
        double GetDaughter1Mass() const {
            return fDaughter1Mass;
        }

        __hydra_host__  __hydra_device__ inline
        void SetDaughter1Mass(double daughter1Mass) {
            fDaughter1Mass = daughter1Mass;
        }

        __hydra_host__  __hydra_device__ inline
        double GetDaughter2Mass() const {
            return fDaughter2Mass;
        }

        __hydra_host__  __hydra_device__ inline
        void SetDaughter2Mass(double daughter2Mass) {
            fDaughter2Mass = daughter2Mass;
        }

        __hydra_host__  __hydra_device__ inline
        double GetDaughter3Mass() const {
            return fBachelorMass;
        }

        __hydra_host__  __hydra_device__ inline
        void SetDaughter3Mass(double daughter3Mass) {
            fBachelorMass = daughter3Mass;
        }

        __hydra_host__  __hydra_device__ inline
        double GetMotherMass() const {
            return fMotherMass;
        }

        __hydra_host__  __hydra_device__ inline
        void SetMotherMass(double motherMass) {
            fMotherMass = motherMass;
        }

        __hydra_host__  __hydra_device__ inline
        double GetRadi() const {
            return fRadi;
        }

        __hydra_host__  __hydra_device__ inline
        void SetRadi(double radi) {
            fRadi = radi;
        }

        template<typename T>
        __hydra_host__ __hydra_device__ inline
        hydra::complex<double> Evaluate(unsigned int, T *x) const {

            const double m = x[ArgIndex];

            const double resonance_mass = _par[0];
            const double rho1 = _par[1];
            const double rho2 = _par[2];

            return m > (fDaughter1Mass + fDaughter2Mass) && m < (fMotherMass - fBachelorMass) ?
                   LineShape(m, resonance_mass, rho1, rho2) : hydra::complex<double>(0.0, 0.0);

        }

        template<typename T>
        __hydra_host__ __hydra_device__ inline
        hydra::complex<double> Evaluate(T x) const {

            double m = get<ArgIndex>(x);

            const double resonance_mass = _par[0];
            const double rho1 = _par[1];
            const double rho2 = _par[2];

            return m > (fDaughter1Mass + fDaughter2Mass) && m < (fMotherMass - fBachelorMass) ?
                   LineShape(m, resonance_mass, rho1, rho2) : hydra::complex<double>(0.0, 0.0);
        }

    private:

        __hydra_dual__ inline
        hydra::complex<double> LineShape(const double s, const double resonance_mass, const double g1 , const double g2) const {

            double pipmass = 0.13957018;
            double pi0mass = 0.1349766;
            double kpmass  = 0.493677;
            double k0mass  = 0.497614;

            double twopimasssq  = 4 * pipmass * pipmass;
            double twopi0masssq = 4 * pi0mass * pi0mass;
            double twokmasssq   = 4 * kpmass * kpmass;
            double twok0masssq  = 4 * k0mass * k0mass;

            double rhokk_real   = 0;
            double rhokk_imag   = 0;
            double rhopipi_real = 0;
            double rhopipi_imag = 0;


            constexpr static double _1p3  = 0.333333333;
            constexpr static double _2p3  = 0.666666667;
            static const double  two_pi0_factor =  ::sqrt(1 - twopi0masssq / s );
            static const double  two_pi_factor  =  ::sqrt(1 - twopimasssq / s );
            static const double  two_pi0_factor_img =  ::sqrt(-1 + twopi0masssq / s );
            static const double  two_pi_factor_img  =  ::sqrt(-1 + twopimasssq / s );

            rhopipi_real = (s >= twopi0masssq)*_1p3*twopi0masssq;

            rhopipi_imag = (s >= twopi0masssq && s < twokmasssq)* (_2p3 * two_pi_factor_img )+
                           (s < twopi0masssq)*_1p3*two_pi0_factor_img;

            constexpr static double _1p2  = 0.5;
            static const double  two_k0_factor =  ::sqrt(1 - twok0masssq / s );
            static const double  two_k_factor  =  ::sqrt(1 - twokmasssq / s );
            static const double  two_k0_factor_img =  ::sqrt(-1 + twok0masssq / s );
            static const double  two_k_factor_img  =  ::sqrt(-1 + twokmasssq / s );


            rhokk_real   = (s >= twokmasssq && s < twok0masssq)*_1p2*two_k_factor +
                           (s >= twok0masssq)*_1p2*two_k0_factor;

            rhokk_imag    = (s < twok0masssq)*_1p2*two_k0_factor_img;


                double A = (resonance_mass*resonance_mass - s) + resonance_mass*(rhopipi_imag*g1 + rhokk_imag*g2);
                double B = resonance_mass*(rhopipi_real*g1 + rhokk_real*g2);
                double C = 1.0 / (A*A + B*B);

                hydra::complex<double> retur(A*C, B*C);

                return retur;


        }

        double fDaughter1Mass;
        double fDaughter2Mass;
        double fBachelorMass;
        double fMotherMass;
        double fRadi;

    };

}
#endif //HYDRA_EXAMPLES_AND_DOCUMENTATION_FLATTELINESHAPE_H
