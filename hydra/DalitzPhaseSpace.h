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
 * DalitzPhaseSpace.h
 *
 *  Created on: 13/12/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DALITZPHASESPACE_H_
#define DALITZPHASESPACE_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/detail/IteratorTraits.h>
#include <hydra/detail/functors/DalitzSampler.h>
#include <hydra/detail/Iterable_traits.h>
#include <hydra/detail/utility/Exception.h>
#include <hydra/Random.h>
#include <array>
#include <utility>
#include <initializer_list>

namespace hydra {

class DalitzWeight
{
	DalitzWeight(double motherMass, const double (&daughtersMasses)[3] )
	{
		fMasses[0]=motherMass;
		fMasses[1]=daughtersMasses[0];
		fMasses[2]=daughtersMasses[1];
		fMasses[2]=daughtersMasses[2];
	}

	__hydra_host__ __hydra_device__
	inline  double operator()(double m12){

		double weight = 1.0;

		weight *= ::sqrt( (m12*m12 - ::pow( fMasses[1]+fMasses[2], 2.0))*
				          (m12*m12 - ::pow( fMasses[1]-fMasses[2], 2.0)) )/(2.0*m12);

		weight *= ::sqrt( ( fMasses[0]*fMasses[0] - ::pow( m12 + fMasses[3], 2.0))*
						  ( fMasses[0]*fMasses[0] - ::pow( m12 - fMasses[3], 2.0)))/(2.0*fMasses[0]);

		return weight;
	}

	double fMasses[4];
};


template <typename GRND=hydra::default_random_engine>
class DalitzPhaseSpace
{

public:

	DalitzPhaseSpace(double motherMass, const double (&daughtersMasses)[3], size_t seed=0xabc123):
    fSeed(seed),
	fMotherMass(motherMass)
    {
		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];
	}

	DalitzPhaseSpace(double motherMass,  std::array<double,3> const& daughtersMasses, size_t seed=0xabc123):
		fSeed(seed),
		fMotherMass(motherMass)
	{
		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];
	}

	DalitzPhaseSpace(double motherMass, std::initializer_list<double> const& daughtersMasses, size_t seed=0xabc123):
		fSeed(seed),
		fMotherMass(motherMass)
	{
		auto ptr = daughtersMasses.begin();
		if(daughtersMasses.size()==3){

			fDaughterMass[0]=*(ptr);
			fDaughterMass[1]=*(ptr+1);
			fDaughterMass[2]=*(ptr+2);
		}
		else
		  HYDRA_EXCEPTION(" Number masses needs to be three." );
	}

	DalitzPhaseSpace( DalitzPhaseSpace<GRND>const& other):
		fSeed(other.GetSeed()),
		fMotherMass(other.GetMotherMass())
	{
		const double* daughtersMasses = other.GetDaughterMass();

		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];
	}

	template<typename GRND2>
	DalitzPhaseSpace( DalitzPhaseSpace<GRND2>const& other):
	fSeed(other.GetSeed()),
	fMotherMass(other.GetMotherMass())
	{
		const double* daughtersMasses = other.GetDaughterMass();

		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];
	}

	DalitzPhaseSpace<GRND>&
	operator=( DalitzPhaseSpace<GRND>const& other)
	{
		if( this == other) return *this;

		fSeed        =other.GetSeed();
		fMotherMass  =other.GetMotherMass();

		const double* daughtersMasses = other.GetDaughterMass();

		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];

		return *this;
	}

	template<typename GRND2>
	DalitzPhaseSpace<GRND>&
	operator=( DalitzPhaseSpace<GRND2>const& other)
	{
		fSeed        =other.GetSeed();
		fMotherMass  =other.GetMotherMass();

		const double* daughtersMasses = other.GetDaughterMass();

		fDaughterMass[0]=daughtersMasses[0];
		fDaughterMass[1]=daughtersMasses[1];
		fDaughterMass[2]=daughtersMasses[2];

		return *this;
	}

	//--------------------

	template<typename FUNCTOR, hydra::detail::Backend BACKEND>
	inline std::pair<GReal_t, GReal_t>
	AverageOn(hydra::detail::BackendPolicy<BACKEND>const& policy, FUNCTOR const& functor, size_t n) ;

    //--------------------

	template< hydra::detail::Backend BACKEND, typename Iterator, typename FUNCTOR >
	inline void Evaluate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy,
			Iterator begin, Iterator end, FUNCTOR const& functor);

	template<typename Iterator, typename FUNCTOR>
	inline void	Evaluate(Iterator begin, Iterator end, FUNCTOR const& functor);

	template<typename Iterable, typename FUNCTOR>
	inline typename std::enable_if< detail::is_iterable<Iterable>::value>::type
	Evaluate(Iterable&& result, FUNCTOR const& functor);

	//--------------------

	template<typename Iterator, typename FUNCTOR>
	inline void Generate( Iterator begin, Iterator end, FUNCTOR const& function);

	template<typename Iterator>
	inline void Generate( Iterator begin, Iterator end);

	template<typename Iterator, hydra::detail::Backend BACKEND>
	inline void Generate(hydra::detail::BackendPolicy<BACKEND> const& exec_policy , Iterator begin, Iterator end);

	template<typename Iterable>
	inline typename std::enable_if<hydra::detail::is_iterable<Iterable>::value>::type
	Generate( Iterable&& events );

	template<typename Iterable, typename FUNCTOR>
	inline typename std::enable_if<
	hydra::detail::is_iterable<Iterable>::value &&
	hydra::detail::random::is_callable<FUNCTOR>::value >::type
	Generate( Iterable&& events, FUNCTOR const& function );

	inline const double* GetDaughterMass() const {
		return fDaughterMass;
	}

	inline void SetDaughterMass(unsigned i , double mass) {
			 fDaughterMass[i]=mass;
	}

	inline double GetMotherMass() const {
		return fMotherMass;
	}

	inline void SetMotherMass(double motherMass) {
		fMotherMass = motherMass;
	}

	inline size_t GetSeed() const {
		return fSeed;
	}

	inline void SetSeed(size_t seed) {
		fSeed = seed;
	}

private:

	template<typename Iterator, typename Functor, typename DerivedPolicy>
	inline void Generate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
			Iterator begin, Iterator end, Functor const& functor);

	template<typename Iterator, typename DerivedPolicy>
		inline void Generate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
				Iterator begin, Iterator end);

	template<typename Iterator, typename FUNCTOR, typename DerivedPolicy>
	inline void Evaluate(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
			Iterator begin, Iterator end, FUNCTOR const& functor);

	template<typename FUNCTOR, typename DerivedPolicy>
	inline std::pair<GReal_t, GReal_t>
	Average(hydra_thrust::detail::execution_policy_base<DerivedPolicy> const& policy,
			FUNCTOR const& functor, size_t n);


	size_t fSeed;
	double fMotherMass;
	double fDaughterMass[3];

};

}  // namespace hydra

#include <hydra/detail/DalitzPhaseSpace.inl>

#endif /* DALITZPHASESPACE_H_ */
