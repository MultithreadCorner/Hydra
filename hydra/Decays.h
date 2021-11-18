/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016 - 2021 Antonio Augusto Alves Junior
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
 * Decays.h
 *
 *  Created on: 24/02/2020
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DECAYS_H_
#define DECAYS_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/multivector.h>
#include <hydra/Tuple.h>
#include <hydra/Function.h>
#include <hydra/PhaseSpace.h>
#include <hydra/detail/FunctorTraits.h>
#include <hydra/detail/CompositeTraits.h>


namespace hydra {

//forward decl
template <typename ...ParticleTypes>
class PhaseSpaceWeight;

//forward decl.
template <typename Functor, typename ...ParticleTypes>
class PhaseSpaceReweight;


/**
* \ingroup phsp
*/
template<typename Particles,  typename Backend>
class Decays;

/**
 * \ingroup phsp
 * \brief This class provides storage for N-particle states. Data is stored using SoA layout.
 * \tparam Particles list of particles in the final state
 * \tparam Backend memory space to allocate storage for the particles.
 */
template<typename ...Particles,   hydra::detail::Backend Backend>
class Decays<hydra::tuple<Particles...>, hydra::detail::BackendPolicy<Backend>>
{
	typedef hydra::detail::BackendPolicy<Backend>  system_type;
	typedef hydra_thrust::tuple<Particles...>       tuple_type;
	typedef multivector<tuple_type, system_type>    storage_type;

	enum { nparticles = sizeof...(Particles) };

public :

	typedef typename storage_type::value_type                                   value_type;
	typedef typename storage_type::reference                                     reference;
	typedef typename storage_type::const_reference                         const_reference;
	typedef typename storage_type::iterator                                       iterator;
	typedef typename storage_type::const_iterator                           const_iterator;
	typedef typename storage_type::reverse_iterator                       reverse_iterator;
	typedef typename storage_type::const_reverse_iterator           const_reverse_iterator;

	/**
	 * Default contstuctor
	 */
	Decays()=delete;

	Decays( double motherMass,  std::array<double, nparticles> const& daughtersMasses, std::size_t nentries=0 ):
		fDecays(nentries),
		fMaxWeight(0.),
	    fMotherMass(motherMass)
	{
		for(std::size_t i=0;i<nparticles;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = fMotherMass;

		for (std::size_t n = 0; n < nparticles; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (std::size_t n = 1; n < nparticles; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= PDK(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}

	Decays( double motherMass, const double (&daughtersMasses)[nparticles], std::size_t nentries=0 ):
		fDecays(nentries),
		fMaxWeight(0.),
	    fMotherMass(motherMass)
	{
		for(std::size_t i=0;i<nparticles;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = fMotherMass;

		for (std::size_t n = 0; n < nparticles; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin = 0.0;
		double wtmax = 1.0;

		for (std::size_t n = 1; n < nparticles; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= PDK(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;
	}

	/**
	 * Copy constructor.
	 * @param other
	 */
	Decays(Decays<tuple_type, system_type> const& other ):
	fDecays(other.GetStorage()),
	fMaxWeight(other.GetMaxWeight()),
    fMotherMass(other.GetMotherMass())
	{
		auto other_masses = other.GetMasses();
		for(std::size_t i=0;i<nparticles;i++)
			fMasses[i]= other_masses[i];

	}
	/**
	 * Move constructor.
	 * @param other
	 */
	Decays(Decays<tuple_type, system_type>&& other ):
		fDecays( other.MoveStorage() ),
		fMaxWeight(other.GetMaxWeight()),
	    fMotherMass(other.GetMotherMass())
	{
		auto other_masses = other.GetMasses();
		for(std::size_t i=0;i<nparticles;i++)
				fMasses[i] = other_masses[i];

	}

	/**
	 * Copy constructor trans-backend
	 * @param other
	 */
	template< hydra::detail::Backend BACKEND2>
	Decays(Decays<tuple_type, detail::BackendPolicy<BACKEND2>> const& other ):
	fDecays(other.GetStorage()),
	fMaxWeight(other.GetMaxWeight()),
    fMotherMass(other.GetMotherMass())
	{
		auto other_masses = other.GetMasses();
		for(std::size_t i=0;i<nparticles;i++)
						fMasses[i]= other_masses[i];
	}

	/**
	 * Copy constructor iterator interface
	 * @param other
	 */
	template<typename Iterator>
	Decays( double motherMass,  std::array<double, nparticles> const& daughtersMasses, Iterator first, Iterator  last ):
	fMaxWeight(0.),
    fMotherMass(motherMass)
	{

		for(std::size_t i=0;i<nparticles;i++)
			fMasses[i] = daughtersMasses[i];

		//compute maximum weight
		double  ECM = fMotherMass;

		for (std::size_t n = 0; n < nparticles; n++)
		{
			ECM -= fMasses[n];
		}

		double emmax = ECM + fMasses[0];
		double emmin  = 0.0;
		double wtmax  = 1.0;

		for (std::size_t n = 1; n < nparticles; n++)
		{
			emmin  += fMasses[n - 1];
			emmax += fMasses[n];
			wtmax *= PDK(emmax, emmin, fMasses[n]);
		}

		fMaxWeight = 1.0 / wtmax;

		std::size_t n = hydra_thrust::distance(first, last);
		fDecays.resize(n);
		hydra_thrust::copy(first, last, fDecays.begin());
	}


	 PhaseSpaceWeight<Particles...>
	 GetEventWeightFunctor() const
	 {
		 return  PhaseSpaceWeight<Particles...>(fMotherMass, fMasses );
	 }

	 template<typename Functor>
	 typename std::enable_if<
	 	 detail::is_hydra_functor<Functor>::value ||
	 	 detail::is_hydra_lambda<Functor>::value  ||
	 	 detail::is_hydra_composite_functor<Functor>::value,
	 	 PhaseSpaceReweight<Functor, Particles...> >::type
	 GetEventWeightFunctor(Functor const& functor) const
	 {
		 return  PhaseSpaceReweight<Functor, Particles...>(functor ,fMotherMass, fMasses );
	 }

	 hydra::Range<iterator>
	 Unweight(std::size_t seed=0x180ec6d33cfd0aba);

	 template<typename Functor>
	 typename std::enable_if<
 	 detail::is_hydra_functor<Functor>::value ||
 	 detail::is_hydra_lambda<Functor>::value  ||
 	 detail::is_hydra_composite_functor<Functor>::value ,
	 hydra::Range<iterator>>::type
	 Unweight( Functor  const& functor, double weight=-1.0, std::size_t seed=0x39abdc4529b1661c);

	/**
	 * Add a decay to the container, increasing
	 * its size by one element.
	 * @param p is a tuple with N final state particles.
	 */
	void AddEntry( tuple_type const& p )
	{
		fDecays.push_back( p );
	}


	reference GetEntry( std::size_t i )
	{
		return fDecays[i];
	}

	template<unsigned int I>
	inline auto	GetDaugtherRange(placeholders::placeholder<I> c)
	-> hydra::Range< decltype ( std::declval< storage_type >().begin(c)) >
	{
		return 	hydra::make_range( fDecays.begin(c), fDecays.end(c) );
	}

	template<typename ...Iterables>
	auto Meld( Iterables&&... iterable)
	-> typename std::enable_if< detail::all_true<detail::is_iterable<Iterables>::value...>::value,
	   decltype(std::declval<storage_type>().meld( std::forward<Iterables>(iterable) ...)) >::type
	{
		return fDecays.meld(std::forward<Iterables>(iterable)... );

	}

	//----------------------------------------
	//  stl compliant interface
	//----------------------------------------
	inline void pop_back()
	{
		fDecays.pop_back();
	}

	inline void push_back(const value_type& particles)
	{
		fDecays.push_back( particles );
	}

	void resize(std::size_t size)
	{
		fDecays.resize(size);
	}

	void clear()
	{
		fDecays.clear();
	}

	void shrink_to_fit()
	{
		fDecays.shrink_to_fit();
	}

	void reserve(std::size_t size)
	{
		fDecays.reserve(size);
	}

	std::size_t size() const
	{
		return fDecays.size();
	}

	std::size_t capacity() const
	{
		return fDecays.capacity();
	}

	bool empty() const
	{
		return fDecays.empty();
	}

	iterator erase(iterator pos)
	{
		return fDecays.erase(pos);
	}

	iterator erase(iterator first, iterator last)
	{
		return fDecays.erase( first, last);
	}

	iterator insert(iterator position, const value_type &x)
	{
		return fDecays.insert(position, x);
	}

	void insert(iterator position, std::size_t n, const value_type &x)
	{
		fDecays.insert(position,n,x );
	}

	template<typename InputIterator>
	void insert(iterator position, InputIterator first, InputIterator last)
	{
		fDecays.insert(position, first, last );
	}

	template<typename Iterable>
	typename std::enable_if<detail::is_iterable<Iterable>::value, void>::type
	insert(iterator position, Iterable range)
	{
		fDecays.insert( position, range);
	}

	reference front()
	{
		return *(fDecays.begin());
	}

	const_reference front() const
	{
		return *(fDecays.cbegin());
	}

	reference back()
	{
		return   fDecays.back();
	}

	const_reference back() const
	{
		return   fDecays .back();
	}

	//converting access
	template<typename Functor>
	auto begin( Functor const& caster )
	-> decltype( std::declval<storage_type>() .begin(caster))
	{
		return fDecays .begin(caster);
	}

	template<typename Functor>
	auto end( Functor const& caster )
	-> decltype( std::declval<storage_type>().end(caster))
	{
		return fDecays.end(caster);
	}

	template<typename Functor>
	auto rbegin( Functor const& caster )
	->decltype( std::declval<storage_type>().rbegin(caster))
	{
		return  fDecays.rbegin(caster);
	}

	template<typename Functor>
	auto rend( Functor const& caster )
	-> decltype( std::declval<storage_type>().rend(caster))
	{
		return fDecays.rend(caster);
	}

	//non-constant access
	iterator begin()
	{
		return  fDecays.begin();
	}

	iterator end()
	{
		return 	fDecays.end();
	}

	reverse_iterator rbegin()
	{
		return 	fDecays.rbegin();
	}

	reverse_iterator rend()
	{
		return 	fDecays.rend();
	}

	template<unsigned int I1, unsigned int ...IN >
	inline  auto begin(placeholders::placeholder<I1> c1, placeholders::placeholder<IN> ...cn)
	-> decltype( std::declval<storage_type>().begin(c1,  cn...) )
	{
		return 	fDecays.begin(c1,  cn...);
	}

	template<unsigned int I1,unsigned int ...IN >
	inline auto	end(placeholders::placeholder<I1> c1,  placeholders::placeholder<IN> ...cn)
	->  decltype( std::declval<storage_type>().end(c1, cn...) )
	{
		return fDecays.end(c1, cn...);
	}





	template<unsigned int I1, unsigned int ...IN >
	inline  auto rbegin(placeholders::placeholder<I1> c1, placeholders::placeholder<IN> ...cn)
	-> decltype( std::declval<storage_type>().rbegin(c1,  cn...) )
	{
		return 	fDecays.rbegin(c1,  cn...);
	}

	template<unsigned int I1,unsigned int ...IN >
	inline auto	rend(placeholders::placeholder<I1> c1,  placeholders::placeholder<IN> ...cn)
	->  decltype( std::declval<storage_type>().rend(c1, cn...) )
	{
		return fDecays.rend(c1, cn...);
	}

	//constant access

	const_iterator begin() const
	{
		return fDecays.begin();
	}

	const_iterator end() const
	{
		return fDecays.end();
	}

	const_reverse_iterator rbegin() const
	{
		return fDecays.rbegin();
	}

	const_reverse_iterator rend() const
	{
		return  fDecays.rend();
	}

	const_iterator cbegin() const
	{
		return fDecays.cbegin();
	}

	const_iterator cend() const
	{
		return fDecays.cend();
	}

	const_reverse_iterator crbegin() const
	{
		return fDecays.crbegin();
	}

	const_reverse_iterator crend() const
	{
		return   fDecays.crend();
	}


	template<unsigned int I1, unsigned int ...IN >
	inline  auto cbegin(placeholders::placeholder<I1> c1, placeholders::placeholder<IN> ...cn) const
	-> decltype( std::declval<storage_type>().cbegin(c1,  cn...) )
	{
		return 	fDecays.cbegin(c1,  cn...);
	}

	template<unsigned int I1,unsigned int ...IN >
	inline auto	cend(placeholders::placeholder<I1> c1,  placeholders::placeholder<IN> ...cn) const
	->  decltype( std::declval<storage_type>().cend(c1, cn...) )
	{
		return fDecays.cend(c1, cn...);
	}

	template<unsigned int I1, unsigned int ...IN >
	inline  auto crbegin(placeholders::placeholder<I1> c1, placeholders::placeholder<IN> ...cn) const
	-> decltype( std::declval<storage_type>().crbegin(c1,  cn...) )
	{
		return 	fDecays.crbegin(c1,  cn...);
	}

	template<unsigned int I1,unsigned int ...IN >
	inline auto	crend(placeholders::placeholder<I1> c1,  placeholders::placeholder<IN> ...cn) const
	->  decltype( std::declval<storage_type>().crend(c1, cn...) )
	{
		return fDecays.crend(c1, cn...);
	}

	//sub-script operators

	inline	reference operator[](std::size_t n)
	{
		return fDecays.begin()[n] ;
	}

	inline const_reference operator[](std::size_t n) const
	{
		return  fDecays.cbegin()[n];
	}

	/**
	 * Assignment operator.
	 * @param other
	 */
	Decays<tuple_type, system_type>&
	operator=(Decays<tuple_type, system_type> const& other )
	{
		if(this==&other) return *this;

		fDecays  = other.GetStoragee();

		return *this;
	}

	/**
	 * Move assignment operator.
	 * @param other
	 * @return
	 */
	Decays<tuple_type, system_type>&
	operator=(Decays<tuple_type, system_type>&& other )
	{
		if(this==&other) return *this;

		this->fDecays  = other.MoveStorage();

		return *this;
	}

const std::array<double,nparticles> GetMasses() const
{
    return fMasses;
}

double GetMaxWeight() const
{
    return fMaxWeight;
}

double GetMotherMass() const
{
    return fMotherMass;
}

	/**
	 * Assignment operator.
	 * @param other
	 * @return
	 */
	template< hydra::detail::Backend BACKEND2>
	Decays<tuple_type, system_type>&
	operator=(Decays<tuple_type,detail::BackendPolicy<BACKEND2> > const& other )
	{

		fDecays.resize( other.size() );

		hydra_thrust::copy(other.begin(),  other.end(), this->begin() );

		return *this;
	}



	const storage_type& GetStorage() const
	{
		return fDecays;
	}

	storage_type&& MoveStorage()
	{
		return std::move(fDecays);
	}


private:

	double PDK(const double a, const double b, const double c) const {
		//the PDK function
		GReal_t x = (a - b - c) * (a + b + c) * (a - b + c) * (a + b - c);
		x = ::sqrt(x) / (2 * a);
		return x;
	}

	storage_type  fDecays;

	double  fMotherMass;
	double  fMaxWeight;
	std::array<double, nparticles>  fMasses;


};


}  // namespace hydra

#include <hydra/detail/Decays.inl>

#endif /* DECAYS_H_ */
