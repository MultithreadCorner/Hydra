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
 * Decays.inl
 *
 *  Created on: 29/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DECAYS_INL_
#define DECAYS_INL_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Vector4R.h>
#include <hydra/Tuple.h>
#include <hydra/Function.h>

namespace hydra {

namespace detail {


template<typename Functor>
class FlagDaugthers
{
public:

	FlagDaugthers()=delete;

	FlagDaugthers(Functor const& functor, double maxWeight, size_t seed=0xd5a61266f0c9392c) :
		fMaxWeight(maxWeight),
		fSeed(seed),
		fFunctor(functor)
        {}

	__hydra_host__  __hydra_device__
	FlagDaugthers(FlagDaugthers<Functor> const&other) :
	fMaxWeight(other.GetMaxWeight()),
	fSeed(other.GetSeed()),
	fFunctor(other.GetFunctor())
	{}

	__hydra_host__  __hydra_device__
	FlagDaugthers<Functor>&
	operator=(FlagDaugthers<Functor> const&other)
	{
		if(this==&other) return *this;

		fMaxWeight = other.GetMaxWeight();
		fSeed      = other.GetSeed();
		fFunctor   = other.GetFunctor();
		return *this;
	}

	template<typename T>
	__hydra_host__  __hydra_device__
	bool operator()(T x) {

		size_t index  = hydra_thrust::get<0>(x);
		double weight = fFunctor( hydra_thrust::get<1>(x) );

		hydra::default_random_engine randEng(fSeed);
		randEng.discard(index);

		hydra_thrust::uniform_real_distribution<double> uniDist(0.0, 1.0);

		return (  weight/fMaxWeight> uniDist(randEng)) ;
	}

	__hydra_host__ __hydra_device__
	const Functor& GetFunctor() const {
		return fFunctor;
	}

	__hydra_host__ __hydra_device__
	double GetMaxWeight() const {
		return fMaxWeight;
	}

	__hydra_host__ __hydra_device__
	size_t GetSeed() const {
		return fSeed;
	}

private:
	size_t  fSeed;
	double  fMaxWeight;
	Functor fFunctor;
};

}  // namespace detail

template <typename ...ParticleTypes>
class PhaseSpaceWeight: public BaseFunctor< PhaseSpaceWeight<ParticleTypes...>, double(ParticleTypes...), 0>
{
	typedef typename detail::signature_type<double,ParticleTypes...>::type  Signature;

	typedef BaseFunctor< PhaseSpaceWeight<ParticleTypes...>, Signature, 0> base_type;

	using base_type::_par;



public:

	PhaseSpaceWeight()=delete;

	PhaseSpaceWeight(double motherMass,  std::array<double,base_type::arity > const& daughtersMasses):
		base_type(),
		fMaxWeight(0.)
	{
		//using  PhaseSpace<base_type::arity>::PDK;

		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = motherMass;

		for (size_t n = 0; n < base_type::arity; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (size_t n = 1; n < base_type::arity; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= pdk(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}

	PhaseSpaceWeight(double motherMass,  const double (&daughtersMasses)[base_type::arity]):
		base_type(),
		fMaxWeight(0.)
	{
		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = motherMass;

		for (size_t n = 0; n < base_type::arity; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (size_t n = 1; n < base_type::arity; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= pdk(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}

	__hydra_host__ __hydra_device__
	PhaseSpaceWeight(PhaseSpaceWeight<ParticleTypes...> const& other ):
	base_type(other),
	fMaxWeight(other.GetMaxWeight())
	{
		for(size_t i=0;i<base_type::arity;i++)
				fMasses[i]= other.GetMasses()[i];
	}


	__hydra_host__ __hydra_device__
	PhaseSpaceWeight<ParticleTypes...> &
	operator=(PhaseSpaceWeight<ParticleTypes...> const& other ){

		if(this==&other) return  *this;

		base_type::operator=(other);
		fMaxWeight=other.GetMaxWeight();

		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i]= other.GetMasses()[i];

		return  *this;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ParticleTypes... p ) const
	{

		hydra::Vector4R particles[base_type::arity]{p...};

	    hydra::Vector4R R = particles[0];
	    double w = fMaxWeight;

	    for(size_t i = 1; i < base_type::arity ; ++i )
	    {
	    	w *= pdk( (R + particles[i]).mass(),  fMasses[i] , R.mass());
	    	R += particles[i];
	    }

	        return w;
	}

	__hydra_host__ __hydra_device__
	const double* GetMasses() const {
		return fMasses;
	}
	__hydra_host__ __hydra_device__
	double GetMaxWeight() const {
		return fMaxWeight;
	}

private:

	__hydra_host__ __hydra_device__
	inline double pdk( double a, double b, double c) const
	{
		//the PDK function
		return ::sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);;
	}

	double  fMaxWeight;
	double  fMasses[base_type::arity];

};



template <typename Functor, typename ...ParticleTypes>
class PhaseSpaceReweight: public BaseFunctor< PhaseSpaceReweight<Functor,ParticleTypes...>, double(ParticleTypes...), 0>
{
	typedef typename detail::signature_type<double,ParticleTypes...>::type  Signature;

	typedef BaseFunctor< PhaseSpaceReweight<Functor,ParticleTypes...>, Signature, 0> base_type;

	using base_type::_par;

public:

	PhaseSpaceReweight()=delete;

	PhaseSpaceReweight(Functor const& functor, double motherMass,  const double (&daughtersMasses)[base_type::arity]):
		base_type(),
		fFunctor(functor),
		fMaxWeight(0.)
	{
		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = motherMass;

		for (size_t n = 0; n < base_type::arity; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (size_t n = 1; n < base_type::arity; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= PDK(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}


	PhaseSpaceReweight(Functor const& functor, double motherMass,  std::array<double,base_type::arity > const& daughtersMasses):
		base_type(),
		fFunctor(functor),
		fMaxWeight(0.)
	{
		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = motherMass;

		for (size_t n = 0; n < base_type::arity; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (size_t n = 1; n < base_type::arity; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= pdk(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}

	__hydra_host__ __hydra_device__
	PhaseSpaceReweight(PhaseSpaceReweight<Functor, ParticleTypes...> const& other ):
	base_type(other),
	fFunctor(other.GetFunctor()),
	fMaxWeight(other.GetMaxWeight())
	{
		for(size_t i=0;i<base_type::arity;i++)
				fMasses[i]= other.GetMasses()[i];
	}


	__hydra_host__ __hydra_device__
	PhaseSpaceReweight<Functor, ParticleTypes...> &
	operator=(PhaseSpaceReweight<Functor, ParticleTypes...> const& other ){

		if(this==&other) return  *this;

		base_type::operator=(other);
		fMaxWeight=other.GetMaxWeight();
		fFunctor     =other.GetFunctor();

		for(size_t i=0;i<base_type::arity;i++)
			fMasses[i]= other.GetMasses()[i];

		return  *this;
	}


	__hydra_host__ __hydra_device__
	inline double Evaluate(ParticleTypes... p ) const
	{

		hydra::Vector4R particles[base_type::arity]{p...};

	    hydra::Vector4R R = particles[0];
	    double w = fMaxWeight;

	    for(size_t i = 1; i < base_type::arity ; ++i )
	    {
	    	w *= pdk( (R + particles[i]).mass(),  fMasses[i] , R.mass());
	    	R += particles[i];
	    }


	        return w*fFunctor(hydra_thrust::make_tuple(p...));
	}

	__hydra_host__ __hydra_device__
	const double* GetMasses() const {
		return fMasses;
	}

	__hydra_host__ __hydra_device__
	double GetMaxWeight() const {
		return fMaxWeight;
	}

	__hydra_host__ __hydra_device__
	 const Functor& GetFunctor() const {
		return fFunctor;
	}


private:

	__hydra_host__ __hydra_device__
	inline double pdk(double a, double b, double c) const
	{
		//the PDK function
		return ::sqrt( (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c) ) / (2 * a);;
	}

	Functor fFunctor;
	double  fMaxWeight;
	double  fMasses[base_type::arity];


};



template<typename ...Particles,   hydra::detail::Backend Backend>
hydra::Range<typename Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>::iterator>
Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>::Unweight(size_t seed)
{
	/*
	 * NOTE: the implementation of this function is not the most efficient in terms
	 * of memory usage. Due probably a bug on thust_stable partition implementation
	 * connected with cuda and tbb, counting iterators can't be deployed as stencil.
	 * So...
	 */

	typedef detail::FlagDaugthers< PhaseSpaceWeight<Particles...> > tagger_type;
	//number of events to trial
	size_t ntrials = fDecays.size();

	//create iterators
	hydra_thrust::counting_iterator < size_t > first(0);
	hydra_thrust::counting_iterator < size_t > last(ntrials);

	auto sequence = hydra_thrust::get_temporary_buffer<size_t>(system_type(), ntrials);
	hydra_thrust::copy(first, last, sequence.first);

	//re-sort the container to build up un-weighted sample
	auto start = hydra_thrust::make_zip_iterator(
			hydra_thrust::make_tuple(sequence.first, fDecays.begin()));

	auto stop = hydra_thrust::make_zip_iterator(
			hydra_thrust::make_tuple(sequence.first + sequence.second, fDecays.end()));

	auto middle = hydra_thrust::stable_partition(start, stop,
			tagger_type(this->GetEventWeightFunctor(), fMaxWeight, seed));

	auto end_of_range = hydra_thrust::distance(start, middle);

	hydra_thrust::return_temporary_buffer(system_type(), sequence.first  );

	//done!
	//return (size_t) hydra_thrust::distance(begin(), middle);
	return hydra::make_range(fDecays.begin(), fDecays.begin() + end_of_range );

}

template<typename ...Particles,   hydra::detail::Backend Backend>
template<typename  Functor>
typename std::enable_if<
 	detail::is_hydra_functor<Functor>::value ||
 	detail::is_hydra_lambda<Functor>::value  ||
 	detail::is_hydra_composite_functor<Functor>::value,
	hydra::Range<typename  Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>::iterator>>::type
Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>::Unweight( Functor  const& functor, double max_weight, size_t seed)
{
/*
	 * NOTE: the implementation of this function is not the most efficient in terms
	 * of memory usage. Due probably a bug on thust_stable partition implementation
	 * connected with cuda and tbb, counting iterators can't be deployed as stencil.
	 * So...
	 */
typedef PhaseSpaceReweight<Functor, Particles...> reweight_functor;
typedef hydra_thrust::transform_iterator<reweight_functor,iterator> reweight_iterator;
typedef detail::FlagDaugthers< reweight_functor> tagger_type;

	//number of events to trial
	size_t ntrials = this->size();

	//create iterators
	hydra_thrust::counting_iterator < size_t > first(0);
	hydra_thrust::counting_iterator < size_t > last(ntrials);

	auto sequence  = hydra_thrust::get_temporary_buffer<size_t>(system_type(), ntrials);
	hydra_thrust::copy(first, last, sequence.first);

	//--------------------

	double max_value = max_weight>0.0 ? max_weight: *(hydra_thrust::max_element(
			reweight_iterator(fDecays.begin(),this->GetEventWeightFunctor(functor) ),
			reweight_iterator(fDecays.end()  ,this->GetEventWeightFunctor(functor) )));


	//re-sort the container to build up un-weighted sample
	auto start  = hydra_thrust::make_zip_iterator(
			hydra_thrust::make_tuple(sequence.first,fDecays.begin()));

	auto stop   = hydra_thrust::make_zip_iterator(
			hydra_thrust::make_tuple(sequence.first + sequence.second,fDecays.end() ));

	auto middle = hydra_thrust::partition(start, stop,
			tagger_type(this->GetEventWeightFunctor(functor), max_value, seed));

	auto end_of_range = hydra_thrust::distance(start, middle);

	hydra_thrust::return_temporary_buffer(system_type(), sequence.first  );

	//done!

	return hydra::make_range(begin(), begin()+end_of_range);

}



}  // namespace hydra

#endif /* DECAYS_INL_ */
