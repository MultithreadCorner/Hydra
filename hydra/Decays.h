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
 * Decays.h
 *
 *  Created on: 04/08/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DECAYS_H_
#define DECAYS_H_

//std
#include <array>
#include <utility>
//hydra
#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/Containers.h>
#include <hydra/Vector3R.h>
#include <hydra/Vector4R.h>
#include <hydra/multiarray.h>
#include <hydra/detail/utility/Utility_Tuple.h>
#include <hydra/Tuple.h>
#include <hydra/Range.h>
#include <hydra/Placeholders.h>
//thrust
#include <hydra/detail/external/thrust/copy.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>
#include <hydra/detail/external/thrust/iterator/reverse_iterator.h>
#include <hydra/detail/external/thrust/iterator/constant_iterator.h>
#include <hydra/detail/external/thrust/iterator/transform_iterator.h>
#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/partition.h>
#include <hydra/detail/external/thrust/random.h>
#include <hydra/detail/external/thrust/extrema.h>
#include <hydra/detail/external/thrust/device_ptr.h>
#include <hydra/detail/external/thrust/detail/type_traits.h>
#include <hydra/detail/external/thrust/tuple.h>

namespace hydra {
/**
* \ingroup phsp
*/
template<size_t N, typename BACKEND>
class Decays;

/**
 * \ingroup phsp
 * \brief This class provides storage for N-particle final states. Data is stored using SoA layout.
 * \tparam N number of particles in the final state
 * \tparam BACKEND memory space to allocate storage for the particles.
 */
template<size_t N, hydra::detail::Backend BACKEND>
class Decays<N, hydra::detail::BackendPolicy<BACKEND> > {

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef HYDRA_EXTERNAL_NS::thrust::tuple<GReal_t,GReal_t, GReal_t, GReal_t> tuple_t;

	typedef multiarray<GReal_t,4,hydra::detail::BackendPolicy<BACKEND>> particles_type;
	typedef std::array<particles_type, N>                               decays_type;
	typedef typename system_t::template container<GReal_t>              weights_type;
	typedef HYDRA_EXTERNAL_NS::thrust::constant_iterator<GReal_t>       unitary_iterator;



	//---------------------------------
	//tuple of direct iterators
	//---------------------------------
    //non const
	typedef typename  detail::tuple_cat_type<
				HYDRA_EXTERNAL_NS::thrust::tuple<typename weights_type::iterator>,
				typename  detail::tuple_type<N,typename particles_type::iterator>::type
			>::type iterator_tuple;
    //const
	typedef typename  detail::tuple_cat_type<
					HYDRA_EXTERNAL_NS::thrust::tuple<typename weights_type::const_iterator>,
					typename  detail::tuple_type<N,typename particles_type::const_iterator>::type
				>::type const_iterator_tuple;

	//accepted events
	typedef typename  detail::tuple_cat_type<
			HYDRA_EXTERNAL_NS::thrust::tuple< unitary_iterator>,
			typename  detail::tuple_type<N,typename particles_type::iterator>::type
			>::type accpeted_iterator_tuple;


	//---------------------------------
	//tuple of reverse iterators
	//---------------------------------
	//non const
	typedef typename  detail::tuple_cat_type<
			HYDRA_EXTERNAL_NS::thrust::tuple<typename weights_type::reverse_iterator>,
			typename  detail::tuple_type<N,typename particles_type::reverse_iterator>::type
			>::type reverse_iterator_tuple;
	//const
	typedef typename  detail::tuple_cat_type<
			HYDRA_EXTERNAL_NS::thrust::tuple<typename weights_type::const_reverse_iterator>,
			typename  detail::tuple_type<N,typename particles_type::const_reverse_iterator>::type
			>::type const_reverse_iterator_tuple;

	//accepted events
	typedef typename  detail::tuple_cat_type<
			HYDRA_EXTERNAL_NS::thrust::tuple< unitary_iterator>,
			typename  detail::tuple_type<N,typename particles_type::reverse_iterator>::type
			>::type accpeted_reverse_iterator_tuple;

public:

	typedef size_t size_type;

	typedef typename detail::tuple_type<N,Vector4R>::type particle_tuple;
	typedef typename  detail::tuple_cat_type<
			HYDRA_EXTERNAL_NS::thrust::tuple< double> , particle_tuple >::type decay_t;
	typedef particle_tuple udecay_t;
	//-----------------------------
	//      stl-like typedefs
	//-----------------------------
	//weights
	//each entry has double type
	typedef typename weights_type::iterator                 weights_iterator;
	typedef typename weights_type::const_iterator           weights_const_iterator;
	typedef typename weights_type::reverse_iterator         weights_reverse_iterator;
	typedef typename weights_type::const_reverse_iterator   weights_const_reverse_iterator;
	typedef typename weights_type::difference_type          weights_difference_type;
	typedef typename weights_type::value_type               weights_value_type;
	typedef typename weights_type::pointer                  weights_pointer;
	typedef typename weights_type::reference                weights_reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<weights_iterator>::iterator_category    weights_iterator_category;

	//particles
	//each entry is [C1, C2, C3, C4 ], C_i has double type
	typedef typename particles_type::iterator                particles_iterator;
	typedef typename particles_type::const_iterator          particles_const_iterator;
	typedef typename particles_type::reverse_iterator        particles_reverse_iterator;
	typedef typename particles_type::const_reverse_iterator  particles_const_reverse_iterator;
	typedef typename particles_type::difference_type         particles_difference_type;
	typedef typename particles_type::value_type              particles_value_type;
	typedef typename particles_type::reference               particles_reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<particles_iterator>::iterator_category    particles_iterator_category;

	//decays
	//each entry is [V1, V2, V3, ..., VN], where V_i has Vector4R type

	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			typename  detail::tuple_type<N,typename particles_type::iterator>::type       >         decays_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			typename  detail::tuple_type<N,typename particles_type::const_iterator>::type >         decays_const_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			typename  detail::tuple_type<N,typename particles_type::reverse_iterator>::type >       decays_reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			typename  detail::tuple_type<N,typename particles_type::const_reverse_iterator>::type > decays_const_reverse_iterator;


	// whole container
	// each entry is [W1, V1, V2, V3, ..., VN ], where V_i has tuple<double, double, double, double> type and W1 is double
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<accpeted_iterator_tuple>      accpeted_iterator;
	//direct
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<iterator_tuple>               iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_iterator_tuple>         const_iterator;
	//reverse
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<reverse_iterator_tuple>       reverse_iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<const_reverse_iterator_tuple> const_reverse_iterator;
	//stl-like typedefs
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::difference_type    difference_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type         value_type;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::pointer            pointer;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference          reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<const_iterator>::reference    const_reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::iterator_category  iterator_category;

	struct __CastTupleToVector4
	{
		__hydra_host__ __hydra_device__
		Vector4R operator()( tuple_t const& v){
			Vector4R r =v; 	return r;
		}
	};


	struct __CastToWeightedDecay
	{
		template<unsigned int I>
		__hydra_host__ __hydra_device__ inline
		typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I==N+1), void >::type
		__convert(value_type const& , decay_t&){ }

		template<unsigned int I=0>
		__hydra_host__ __hydra_device__ inline
		typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I<N+1), void >::type
		__convert(value_type const& v , decay_t& r)
		{
			HYDRA_EXTERNAL_NS::thrust::get<I>(r) =
					HYDRA_EXTERNAL_NS::thrust::get<I>(v);
			__convert<I+1>(v, r );
		}

		__hydra_host__ __hydra_device__ inline
		decay_t operator()( value_type const& v){
			decay_t r{}; __convert(v, r ); 	return r;
		}

	};



	struct __CastToUnWeightedDecay
	{
		template<unsigned int I>
		__hydra_host__ __hydra_device__ inline
		typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I==N), void >::type
		__convert_helper(value_type& , udecay_t& ){ }

		template<unsigned int I=0>
		__hydra_host__ __hydra_device__ inline
		typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I<N), void >::type
		__convert_helper(value_type& v , udecay_t& r)
		{
			HYDRA_EXTERNAL_NS::thrust::get<I>(r) = HYDRA_EXTERNAL_NS::thrust::get<I+1>(v);
			__convert_helper<I+1>(v, r );
		}
		__hydra_host__ __hydra_device__ inline
		void __convert(value_type & v , udecay_t& r)
		{
			__convert_helper(v, r );
		}
		__hydra_host__ __hydra_device__ inline
		udecay_t operator()( value_type & v){
			udecay_t r;
			__convert( v , r);
			return r;
		}

	};


	//cast iterator
	template < typename Iterator, typename Arg, typename Functor>
	using __caster_iterator = HYDRA_EXTERNAL_NS::thrust::transform_iterator<Functor,
			Iterator, typename std::result_of<Functor(Arg&)>::type >;

	/**
	 * Default contstuctor
	 */
	Decays():
		fDecays(decays_type()),
		fWeights(weights_type())
	{};

	/**
	 * Constructor with n decays.
	 * @param n
	 */
	Decays(size_t n):
		fDecays(),
		fWeights(n)
	{
		for( size_t i=0; i<N; i++)
			fDecays[i].resize(n);
	};

	/**
	 * Copy constructor.
	 * @param other
	 */
	Decays(Decays<N,detail::BackendPolicy<BACKEND>> const& other ):
		fDecays(other.__copy_decays()),
		fWeights(other.__copy_weights())
	{ }

	/**
	 * Move constructor.
	 * @param other
	 */
	Decays(Decays<N,detail::BackendPolicy<BACKEND>>&& other ):
		fDecays(other.__move_decays()),
		fWeights(other.__move_weights())
	{}

	/**
	 * Assignment operator.
	 * @param other
	 */
	template< hydra::detail::Backend BACKEND2>
	Decays(Decays<N,detail::BackendPolicy<BACKEND2>> const& other )
	{
		this->resize(HYDRA_EXTERNAL_NS::thrust::distance(other.begin(),  other.end()));
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(),  other.end(), this->begin() );
	}

	/**
	 * Assignment operator.
	 * @param other
	 */
	template<typename Iterator>
	Decays( Iterator first, Iterator  last )
	{
		size_t n = HYDRA_EXTERNAL_NS::thrust::distance(first, last);

		for( size_t i=0; i<N; i++)
			fDecays[i].resize(n);
		fWeights.resize(n);

		HYDRA_EXTERNAL_NS::thrust::copy(first, last, this->begin());
	}

	/**
	 * Assignment operator.
	 * @param other
	 */
	Decays<N,detail::BackendPolicy<BACKEND>>&
	operator=(Decays<N,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(this==&other) return *this;
		this->fDecays  = other.__copy_decays();
		this->fWeights = other.__copy_weights();

		return *this;
	}

	/**
	 * Move assignment operator.
	 * @param other
	 * @return
	 */
	Decays<N,detail::BackendPolicy<BACKEND>>&
	operator=(Decays<N,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(this==&other) return *this;
		this->fDecays  = other.__move_decays();
		this->fWeights = other.__move_weights();
		return *this;
	}

	/**
	 * Assignment operator.
	 * @param other
	 * @return
	 */
	template< hydra::detail::Backend BACKEND2>
	Decays<N,detail::BackendPolicy<BACKEND> >&
	operator=(Decays<N,detail::BackendPolicy<BACKEND2> > const& other )
	{
		HYDRA_EXTERNAL_NS::thrust::copy(other.begin(),  other.end(), this->begin() );

		return *this;
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is a tuple with N final state particles.
	 */
	void AddDecay(GReal_t w, const particle_tuple& p ){

		this->fWeights.push_back(w);
		__push_back(p);

	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is an array with N final state particles.
	 */
	void AddDecay(GReal_t w, Vector4R const (&p)[N]){

		this->fWeights.push_back(w);
		__push_back(p);

	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is a braced list with N final state particles.
	 */
	void AddDecay( GReal_t w, std::array<Vector4R, N> const& p){

		this->fWeights.push_back(w);
		__push_back( p);
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param value is a hydra::tuple<double, Vector4R,...,Vector4R >
	 *  = {weight, \f$p_1,...,p_N\f$}.
	 */
	void AddDecay(value_type const& value){

		__push_back(value);
	}

	/**
	 * Get the range containing the particle number \f$i\f$.
	 * @param i index of particle.
	 * @return std::pair of iterators {begin, end}.
	 */
	Range< __caster_iterator< typename particles_type::iterator, tuple_t ,  __CastTupleToVector4> >
	GetParticles(size_t i){

		return hydra::make_range(this->fDecays[i].begin(__CastTupleToVector4()),
					this->fDecays[i].end(__CastTupleToVector4()));
	}

	Range<weights_iterator >
	GetWeights(){

			return hydra::make_range(this->fWeights.begin(),
						this->fWeights.end());
		}

	Range<weights_const_iterator >
	GetWeights() const {

		return hydra::make_range(this->fWeights.begin(),
				this->fWeights.end());
	}

	Range<typename particles_type::iterator >
	GetDaughters(size_t i){

			return hydra::make_range(this->fDecays[i].begin(),
						this->fDecays[i].end());
		}

	Range<typename particles_type::const_iterator >
	GetDaughters(size_t i) const {

		return hydra::make_range(this->fDecays[i].begin(),
				this->fDecays[i].end());
	}

	/**
	 * Get a constant reference to the internal vector holding the particle i.
	 * Users can not resize or change the hold values. This method is most useful
	 * in python bindings to avoid exposition of iterators.
	 * @param i  index of the particle.
	 * @return reference to constant  particles_type.
	 */
	const particles_type&  GetListOfParticles(size_t i) const {
		return fDecays[i];
	}

	/**
	 * Get the range containing the component particle number \f$i\f$.
	 * @param i Particle index.
	 * @param j Component index
	 * @return std::pair of iterators {begin, end}.
	 */
	Range<typename particles_type::iterator_v >
	GetParticleComponents(size_t i, size_t j){

		return hydra::make_range(this->fDecays[i].begin(j),
				this->fDecays[i].end(j));
	}

	/**
	 * Get a constant reference to the internal vector holding the component j of the particle i.
	 * Users can not resize or change the hold values. This method is most useful
	 * in python bindings to avoid exposition of iterators.
	 * @param i index of the particle.
	 * @param j index of the component.
	 * @return reference to constant  column_type.
	 */
	const typename particles_type::column_type&
	GetListOfParticleComponents(size_t i, size_t j){
		return fDecays[i].column(j);
	}

	/**
	 * Get a reference a decay.
	 * @param i index of the decay.
	 * @return reference a decay
	 */

	 decay_t GetDecay(size_t i){
		return this->begin( __CastToWeightedDecay())[i];
	}


	Range< __caster_iterator<iterator,  value_type, __CastToUnWeightedDecay > >
	GetUnweightedDecays(){

		return make_range(this->begin(__CastToUnWeightedDecay()),
				    this->end(__CastToUnWeightedDecay()));
	}

	Range<  __caster_iterator<iterator,  value_type, __CastToWeightedDecay > >
	GetWeightedDecays(){

		return make_range(this->begin(__CastToWeightedDecay()),
					this->end( __CastToWeightedDecay()));
	}

	/**
	 * Get a constant reference a decay.
	 * @param i index of the decay.
	 * @return reference a decay
	 */
	void SetDecay( size_t i,  decay_t value) {
		this->begin()[i]= value;
	}

	/**
	 * Get a range pointing to a set of unweighted events.
	 * This method will re-order the container to group together
	 * accepted events and return the index of the last event.
	 *
	 * @return index of last unweighted event.
	 */
	size_t Unweight(GUInt_t scale=1.0);

	/**
	 * Get a range pointing to a set of unweighted events.
	 * This method will re-order the container to group together
	 * accepted events and return a pair of iterators with ready-only
	 * access to container. This version takes a functor as argument
	 * and will produce a range of unweighted events distributed
	 * accordingly. This method does not change the size, the
	 * stored events or its weights.
	 * The functor needs derive from hydra::BaseFunctor or be a lambda wrapped
	 * using hydra::wrap_lambda function.
	 * The functor signature needs to provide
	 * the method Evaluate(size_t n, hydra::Vector4R*), for example:
	 * @code{.cpp}
	 * ...
	 * struct BreitWigner: public hydra::BaseFunctor<BreitWigner, double, 0>
	 * ...
	 * double Evaluate(size_t n, hydra::Vector4R* particles)
	 * {
	 *   Vector4R p1 = particles[0];
	 *   Vector4R p2 = particles[1];
	 *
	 *   return  breit_wigner(p1,p2);
	 * }
	 * ...
	 * };
	 * @endcode
	 * The same is is valid for a lambda function:
	 *
	 * @code{.cpp}
	 * ...
	 *
	 * double mass  = ...;
	 * double width = ...;
	 *
	 * auto bw = [ ]__hydra_host__ __hydra_device__(size_t n, hydra::Vector4R* particles )
	 * {
	 * auto   p0  = particles[0] ;
	 * auto   p1  = particles[1] ;
	 * auto   p2  = particles[2] ;
	 *
	 * auto   m = (p1+p2).mass();
	 *
	 * double denominator = (m12-0.895)*(m12-0.895) + (0.055*0.055)/4.0;
	 *
	 * return ((0.055*0.055)/4.0)/denominator;
	 *
	 * };
	 *
	 *auto breit_wigner = hydra::wrap_lambda(bw);
	 *@endcode
	 *
	 * Obs: the functor need be positive evaluated for all events
	 * in the phase-space.
	 *
	 * @tparam Functor enclosing a positive evaluated function.
	 * @return std::pair with iterators pointing to a range of unweigted
	 * particles.
	 */
	template<typename FUNCTOR>
	size_t Unweight( FUNCTOR  const& functor, GUInt_t scale);

	/**
	 * Recalculates the events weights according with @functor;
	 * The new weights are the \f$ w_{i}^{new} = w_{i}^{old} \times functor(Vector4R*...)\f$
	 * * The functor needs derive from hydra::BaseFunctor or be a lambda wrapped
	 * using hydra::wrap_lambda function.
	 * The functor signature needs to provide
	 * the method Evaluate(size_t n, hydra::Vector4R*), for example:
	 * @code{.cpp}
	 * ...
	 * struct BreitWigner: public hydra::BaseFunctor<BreitWigner, double, 0>
	 * ...
	 * double Evaluate(size_t n, hydra::Vector4R* particles)
	 * {
	 *   Vector4R p1 = particles[0];
	 *   Vector4R p2 = particles[1];
	 *
	 *   return  breit_wigner(p1,p2);
	 * }
	 * ...
	 * };
	 * @endcode
	 * The same is is valid for a lambda function:
	 *
	 * @code{.cpp}
	 * ...
	 *
	 * double mass  = ...;
	 * double width = ...;
	 *
	 * auto bw = [ ]__hydra_host__ __hydra_device__(size_t n, hydra::Vector4R* particles )
	 * {
	 * auto   p0  = particles[0] ;
	 * auto   p1  = particles[1] ;
	 * auto   p2  = particles[2] ;
	 *
	 * auto   m = (p1+p2).mass();
	 *
	 * double denominator = (m12-0.895)*(m12-0.895) + (0.055*0.055)/4.0;
	 *
	 * return ((0.055*0.055)/4.0)/denominator;
	 *
	 * };
	 *
	 *auto breit_wigner = hydra::wrap_lambda(bw);
	 *@endcode
	 *
	 * Obs: the functor need be positive evaluated for all events
	 * in the phase-space.
	 *
	 * @tparam Functor enclosing a positive evaluated function.
	 * particles.
	 * @param functor
	 */
	template<typename FUNCTOR>
	void Reweight( FUNCTOR  const& functor);


	//stl compliant interface
	//-----------------------
	inline void pop_back(){
		this->__pop_back();
	}

	inline void push_back(GReal_t weight, const particle_tuple& particles){

		this->fWeights.push_back( weight );
		this->__push_back( particles );
	}

	inline void push_back(GReal_t weight, Vector4R const (&particles)[N]){

		this->fWeights.push_back( weight );
		this->__push_back( particles );
	}


	inline void	push_back(value_type const& value){ this->__push_back( value ); }

	void resize(size_t size){ __resize(size); }

	void clear(){__clear(); }

	void shrink_to_fit() { __shrink_to_fit(); }

	void reserve(size_t size) { __reserve(size); }

	size_t size() const{return this->fWeights.size(); }

	size_t capacity() const{return this->fWeights.capacity();}

	bool empty() const{	return this->fWeights.empty(); }

	iterator erase(iterator pos) {
		size_t n = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), pos);
		__erase( n );
		return this->begin() + n;
	}

	iterator erase(iterator first, iterator last){

		size_t _first = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), first);
		size_t _last  = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), last);

		__erase(_first,  _last  );
		return this->begin() + _first;
	}

	iterator insert(iterator position, const value_type &x){
		size_t _pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
		__insert(_pos, x);

		return this->begin() + _pos;
	}

	void insert(iterator position, size_type n, const value_type &x){

		size_t _pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
		__insert(_pos, n, x);

	}


	template<typename InputIterator>
	void insert(iterator position, InputIterator first, InputIterator last){
		auto _first = first.get_iterator_tuple();
		auto _last  = last.get_iterator_tuple();
		size_t _pos = HYDRA_EXTERNAL_NS::thrust::distance(this->begin(), position);
		__insert( _pos, _first, _last );

	}

	reference front(){	return this->begin()[0];}

	const_reference front() const  {return this->cbegin()[0];}

	reference back(){return  this->begin()[this->size() - 1];}

	const_reference back() const{return  this->cbegin()[this->size() - 1]; }

	//converting access
	template<typename Functor>
	__caster_iterator<iterator, value_type, Functor>
	begin( Functor const& caster )
	{ return __begin(caster);}

	template<typename Functor>
	__caster_iterator<iterator, value_type, Functor>
	end( Functor const& caster )
	{ return __end(caster);}

	template<typename Functor>
	__caster_iterator<reverse_iterator, value_type, Functor>
	rbegin( Functor const& caster )
	{ return __rbegin(caster);}

	template<typename Functor>
	__caster_iterator<reverse_iterator, value_type, Functor>
	rend( Functor const& caster )
	{ return __rend(caster);}

	//non-constant access
	iterator begin(){ return __begin(); }

	iterator end(){ return __end(); }

	reverse_iterator rbegin(){ return __rbegin(); }

	reverse_iterator rend(){ return __rend(); }

	//constant access
	const_iterator begin() const {return __begin();}

	const_iterator end() const { return __end(); }

	const_reverse_iterator rbegin() const { return __rbegin();}

	const_reverse_iterator rend() const { return __rend();}

	const_iterator cbegin() const { return __cbegin(); }

	const_iterator cend() const { return __cend(); }

	const_reverse_iterator crbegin() const { return __crbegin(); }

	const_reverse_iterator crend() const { return  __crend();}

	inline	reference operator[](size_t n)
	{	return begin()[n] ;	}

	inline const_reference operator[](size_t n) const
	{	return cbegin()[n]; }



private:

	const weights_type& __copy_weights() const { return fWeights; }
	const  decays_type& __copy_decays() const { return fDecays; }

	weights_type __move_weights() { return std::move(fWeights); }
	decays_type  __move_decays() { return std::move(fDecays); }

	//_______________________________________________
	//pop_back

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__pop_back_helper(){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__pop_back_helper()
	{
		std::get<I>(fDecays).pop_back();
		__pop_back_helper<I+1>();
	}

	void __pop_back() { fWeights.pop_back(); __pop_back_helper(); }

	//_______________________________________________
	//resize

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__resize_helper( size_t ){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__resize_helper( size_t n)
	{
		std::get<I>(fDecays).resize(n);
		__resize_helper<I+1>(n);
	}

	void __resize( size_t n) { fWeights.resize(n); __resize_helper(n); }

	//_______________________________________________
	//clear

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__clear_helper(){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__clear_helper()
	{
		std::get<I>(fDecays).clear();
		__clear_helper<I+1>();
	}

	void __clear() { fWeights.clear(); __clear_helper(); }

	//_______________________________________________
	//shrink_to_fit

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__shrink_to_fit_helper(){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__shrink_to_fit_helper()
	{
		std::get<I>(fDecays).shrink_to_fit();
		__shrink_to_fit_helper<I+1>();
	}

	void __shrink_to_fit() { fWeights.shrink_to_fit(); __shrink_to_fit_helper(); }

	//_______________________________________________
	//reserve

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__reserve_helper( size_t ){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__reserve_helper( size_t n)
	{
		std::get<I>(fDecays).reserve(n);
		__reserve_helper<I+1>(n);
	}

	void __reserve( size_t n) { fWeights.reserve(n); __reserve_helper(n); }

	//_______________________________________________
	//insert
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper(size_type , size_type, value_type const& ){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper(size_type i, size_type n, value_type const& x )
	{
		std::get<I>(fDecays).insert( HYDRA_EXTERNAL_NS::thrust::get<I>(fDecays).begin() + i, n,
				HYDRA_EXTERNAL_NS::thrust::get<I+1>(x)  ); ;

		__insert_helper<I+1>(i, n, x);
	}

	void __insert( size_type i, size_type n, value_type const& x ) {

		fWeights.insert( fWeights.begin()+i, n, HYDRA_EXTERNAL_NS::thrust::get<0>(x)  );
		__insert_helper(i, n, x);

	}

	//----
	template<size_t I, typename ...Iterators >
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper( size_type , HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...>,
			HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...>  ) {	}

	template< size_t I =0, typename ...Iterators >
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper( size_type pos, HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...> const& first,
			HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...>const& last ) {

		std::get<I>(fDecays).insert( std::get<I>(fDecays).begin()+pos,
				HYDRA_EXTERNAL_NS::thrust::get<I+1>(first),
				HYDRA_EXTERNAL_NS::thrust::get<I+1>(last));

		__insert_helper<I+1>(pos, first, last);
	}

	template<typename ...Iterators>
	void __insert( size_type pos, HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...> first,
			HYDRA_EXTERNAL_NS::thrust::tuple<Iterators...> last ) {

		fWeights.insert( fWeights.begin()+pos, HYDRA_EXTERNAL_NS::thrust::get<0>(first),
				HYDRA_EXTERNAL_NS::thrust::get<0>(last));

		__insert_helper(pos, first, last);
	}

	//----
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__insert_helper(size_type , value_type const& ){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__insert_helper(size_type pos, value_type const& x )
	{
		std::get<I>(fDecays).insert( HYDRA_EXTERNAL_NS::thrust::get<I>(fDecays).begin() + pos,
				HYDRA_EXTERNAL_NS::thrust::get<I+1>(x)  ); ;

		__insert_helper<I+1>(pos, x);
	}

	void __insert( size_type pos, value_type const& x ) {

		fWeights.insert( fWeights.begin()+pos, HYDRA_EXTERNAL_NS::thrust::get<0>(x)  );
		__insert_helper(pos, x);
	}

	//_______________________________________________
	//erase
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__erase_helper(size_type){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__erase_helper(size_type pos )
	{
		std::get<I>(fDecays).erase( HYDRA_EXTERNAL_NS::thrust::get<I>(fDecays).begin() + pos);

		__erase_helper<I+1>(pos);
	}

	void __erase( size_type pos ) {

		fWeights.erase( fWeights.begin()+pos );
		__erase_helper(pos);
	}

	//-----

	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__erase_helper(size_type, size_type){ }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__erase_helper(size_type first, size_type last)
	{
		std::get<I>(fDecays).erase( std::get<I>(fDecays).begin() + first,
				std::get<I>(fDecays).begin() + last );

		__erase_helper<I+1>(first, last);
	}

	void __erase( size_type first, size_type last ) {

		fWeights.erase( fWeights.begin()+first,  fWeights.begin()+ last);
		__erase_helper(first, last );
	}


	//_______________________________________________
	//push_back
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back(const particle_tuple&){	}

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back( const particle_tuple& p )
	{
		fDecays[I].push_back( HYDRA_EXTERNAL_NS::thrust::get<I>(p) );
		__push_back<I+1>(p );
	}

	//----
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back(Vector4R const (&p)[N]){ (void)p; }

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back(Vector4R const (&p)[N])
	{
		fDecays[I].push_back( p[I] );
		__push_back<I+1>(p);
	}

	//----
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back(std::array<Vector4R, N> const& ){ 	}

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back(std::array<Vector4R, N> const& p)
	{
		fDecays[I].push_back( p[I] );
		__push_back<I+1>(p);
	}

	//---
	template<size_t I>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I == N), void >::type
	__push_back_helper(value_type const& ){ 	}

	template<size_t I = 0>
	inline typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if<(I < N), void >::type
	__push_back_helper(value_type const& p)
	{
		fDecays[I].push_back( HYDRA_EXTERNAL_NS::thrust::get<I>(p)  );
		__push_back_helper<I+1>(p);
	}

	void __push_back(value_type const& p) {

		fWeights.push_back(  HYDRA_EXTERNAL_NS::thrust::get<0>(p)  );
		__push_back_helper(p);
	}

	//__________________________________________
	// caster accessors

	template<typename Functor>
	 inline __caster_iterator<iterator, value_type, Functor> __begin( Functor const& caster )
	{
		return __caster_iterator<iterator, value_type, Functor>(this->begin(), caster);
	}

	template<typename Functor>
	 inline __caster_iterator<iterator, value_type, Functor> __end( Functor const& caster )
	{
		return __caster_iterator<iterator, value_type, Functor>(this->end(), caster);
	}

	template<typename Functor>
	 inline __caster_iterator<reverse_iterator, value_type, Functor> __rbegin( Functor const& caster )
	{
		return __caster_iterator<reverse_iterator, value_type, Functor>(this->rbegin(), caster);
	}

	template<typename Functor>
	inline __caster_iterator<reverse_iterator, value_type, Functor> __rend( Functor const& caster )
	{
		return __caster_iterator<reverse_iterator, value_type, Functor>(this->end(), caster);
	}

	// non-constant access

	//___________________________________________
	//begin
	template < size_t... I>
	iterator __begin_helper(detail::index_sequence<I...>){

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
					HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin(),  std::get<I>(fDecays).begin() ...));
	}

	iterator __begin(){	return __begin_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//end
	template < size_t... I>
	iterator __end_helper(detail::index_sequence<I...>){

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end(), std::get<I>(fDecays).end()... ));
	}

	iterator __end(){	return __end_helper( detail::make_index_sequence<N>{} );}
	//___________________________________________
	//rbegin
	template < size_t... I>
	reverse_iterator __rbegin_helper(detail::index_sequence<I...>){

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.rbegin(),  std::get<I>(fDecays).rbegin()... ));
	}

	reverse_iterator __rbegin(){	return __rbegin_helper( detail::make_index_sequence<N>{} );}
	//___________________________________________
	//rend
	template < size_t... I>
	reverse_iterator __rend_helper(detail::index_sequence<I...>){

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.rend(), std::get<I>(fDecays).rend() ...));
	}

	reverse_iterator __rend(){	return __rend_helper( detail::make_index_sequence<N>{} );}

	// constant access
	//___________________________________________
	//begin
	template < size_t... I>
	const_iterator __begin_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.begin(),  std::get<I>(fDecays).begin() ...));
	}

	const_iterator __begin() const {	return __begin_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//end
	template < size_t... I>
	const_iterator __end_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.end(), std::get<I>(fDecays).end() ...));
	}

	const_iterator __end() const {	return __end_helper( detail::make_index_sequence<N>{} );}
	//___________________________________________
	//rbegin
	template < size_t... I>
	const_reverse_iterator __rbegin_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.rbegin(),  std::get<I>(fDecays).rbegin() ...));
	}

	const_reverse_iterator __rbegin() const {	return __rbegin_helper( detail::make_index_sequence<N>{} );}
	//___________________________________________
	//rend
	template < size_t... I>
	const_reverse_iterator __rend_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.rend(), std::get<I>(fDecays).rend() ...));
	}

	const_reverse_iterator __rend() const {	return __rend_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//begin
	template < size_t... I>
	const_iterator __cbegin_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cbegin(),  std::get<I>(fDecays).cbegin() ...));
	}

	const_iterator __cbegin() const {	return __cbegin_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//end
	template < size_t... I>
	const_iterator __cend_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.cend(), std::get<I>(fDecays).cend() ...));
	}

	const_iterator __cend() const {	return __cend_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//rbegin
	template < size_t... I>
	const_reverse_iterator __crbegin_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.crbegin(),  std::get<I>(fDecays).crbegin()... ));
	}

	const_reverse_iterator __crbegin() const {	return __crbegin_helper( detail::make_index_sequence<N>{} );}
	//___________________________________________
	//rend
	template < size_t... I>
	const_reverse_iterator __crend_helper(detail::index_sequence<I...>) const {

		return  HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fWeights.crend(), std::get<I>(fDecays).crend() ...));
	}

	const_reverse_iterator __crend() const {	return __crend_helper( detail::make_index_sequence<N>{} );}

	//___________________________________________
	//


	decays_type  fDecays;
	weights_type fWeights;

};

template<size_t N, hydra::detail::Backend BACKEND>
Decays<N, hydra::detail::BackendPolicy<BACKEND> >
make_decays( hydra::detail::BackendPolicy<BACKEND>, size_t entries ){

	return Decays<N, hydra::detail::BackendPolicy<BACKEND> >(entries);
}

template<size_t N1, hydra::detail::Backend BACKEND1,
         size_t N2, hydra::detail::Backend BACKEND2>
bool operator==(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs);

template<size_t N1, hydra::detail::Backend BACKEND1,
         size_t N2, hydra::detail::Backend BACKEND2>
bool operator!=(const Decays<N1, hydra::detail::BackendPolicy<BACKEND1> >& lhs,
                const Decays<N2, hydra::detail::BackendPolicy<BACKEND2> >& rhs);
}  // namespace hydra


#include <hydra/detail/Decays.inl>

#endif /* DECAYS_H_ */
