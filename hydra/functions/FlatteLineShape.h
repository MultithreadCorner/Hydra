//
// Created by juan on 07/03/18.
//

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
                fDaughter3Mass(daugther3_mass),
                fMotherMass(mother_mass),
                fRadi(radi) {}

        __hydra_host__  __hydra_device__
        FlatteLineShape(FlatteLineShape<CHANNEL, ArgIndex> const &other) :
                BaseFunctor<FlatteLineShape<CHANNEL, ArgIndex>, hydra::complex<double>, 3>(other),
                fDaughter1Mass(other.GetDaughter1Mass()),
                fDaughter2Mass(other.GetDaughter2Mass()),
                fDaughter3Mass(other.GetDaughter3Mass()),
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
            fDaughter3Mass = other.GetDaughter3Mass();
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
            return fDaughter3Mass;
        }

        __hydra_host__  __hydra_device__ inline
        void SetDaughter3Mass(double daughter3Mass) {
            fDaughter3Mass = daughter3Mass;
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

            return m > (fDaughter1Mass + fDaughter2Mass) && m < (fMotherMass - fDaughter3Mass) ?
                   LineShape(m, resonance_mass, rho1, rho2) : hydra::complex<double>(0.0, 0.0);

        }

        template<typename T>
        __hydra_host__ __hydra_device__ inline
        hydra::complex<double> Evaluate(T x) const {

            double m = get<ArgIndex>(x);

            const double resonance_mass = _par[0];
            const double rho1 = _par[1];
            const double rho2 = _par[2];

            return m > (fDaughter1Mass + fDaughter2Mass) && m < (fMotherMass - fDaughter3Mass) ?
                   LineShape(m, resonance_mass, rho1, rho2) : hydra::complex<double>(0.0, 0.0);
        }

    private:


        __hydra_host__ __hydra_device__ inline
        hydra::complex<double>
        LineShape(const double s, const double resonance_mass, const double g1 , const double g2) const {

            hydra::complex<double> ret(0.0,0.0);

            double pipmass = 0.13957018;
            double pi0mass = 0.1349766;
            double kpmass  = 0.493677;
            double k0mass  = 0.497614;

            double twopimasssq  = 4 * pipmass * pipmass;
            double twopi0masssq = 4 * pi0mass * pi0mass;
            double twokmasssq   = 4 * kpmass * kpmass;
            double twok0masssq  = 4 * k0mass * k0mass;

                double rhopipi_real = 0, rhopipi_imag = 0;
                double rhokk_real = 0, rhokk_imag = 0;


                if(s >= twopimasssq)
                    rhopipi_real = (2. / 3) * TMath::Sqrt(1 - twopimasssq / s ); // Above pi+pi- threshold
                else
                    rhopipi_imag = (2. / 3) * TMath::Sqrt(-1 + twopimasssq / s);
                if(s >= twopi0masssq)
                    rhopipi_real = (1. / 3) * TMath::Sqrt(1 - twopi0masssq / s ); // Above pi0pi0 threshold
                else
                    rhopipi_imag = (1. / 3) * TMath::Sqrt(-1 + twopi0masssq / s );
                if(s >= twokmasssq)
                    rhokk_real = 0.5 * TMath::Sqrt(1 - twokmasssq / s ); // Above K+K- threshold
                else
                    rhokk_imag = 0.5 * TMath::Sqrt(-1 + twokmasssq / s );
                if(s >= twok0masssq)
                    rhokk_real = 0.5 * TMath::Sqrt(1 - twok0masssq / s ); // Above K0K0 threshold
                else
                    rhokk_imag = 0.5 * TMath::Sqrt(-1 + twok0masssq / s );

                double A = (resonance_mass * resonance_mass - s) + resonance_mass * (rhopipi_imag * g1 + rhokk_imag * g2);
                double B = resonance_mass * (rhopipi_real * g1 + rhokk_real * g2);
                double C = 1.0 / (A * A + B * B);

                hydra::complex<double> retur(A * C, B * C);


                ret+=retur;

            return ret;


        }

        double fDaughter1Mass;
        double fDaughter2Mass;
        double fDaughter3Mass;
        double fMotherMass;
        double fRadi;

    };

}
#endif //HYDRA_EXAMPLES_AND_DOCUMENTATION_FLATTELINESHAPE_H
