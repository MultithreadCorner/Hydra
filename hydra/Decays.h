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
//thrust
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/partition.h>
#include <thrust/random.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

namespace hydra {


template<size_t N, typename BACKEND>
class Decays;

template<size_t N, hydra::detail::Backend BACKEND>
class Decays<N, hydra::detail::BackendPolicy<BACKEND> > {

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef multiarray<4,GReal_t, hydra::detail::BackendPolicy<BACKEND> > particles_type;
	typedef std::array<particles_type, N> decays_type;
	typedef typename system_t::template container<GReal_t> weights_type;
	typedef thrust::constant_iterator<GReal_t>      weight_one_iterator;

	//pointers

	typedef typename  detail::tuple_type<N,typename particles_type::pointer_tuple>::type particles_pointer_tuple_type;
	typedef typename  thrust::tuple<typename weights_type::pointer> weights_pointer_type;

	typedef typename  detail::tuple_type<N,typename particles_type::const_pointer_tuple>::type particles_const_pointer_tuple_type;
	typedef typename  thrust::tuple<typename weights_type::const_pointer> weights_const_pointer_type;


	//direct iterators

	typedef typename  detail::tuple_type<N,typename particles_type::iterator>::type tuple_particles_iterator_type;
	typedef typename  thrust::tuple<typename weights_type::iterator> tuple_weights_iterator_type;
	typedef typename  detail::tuple_cat_type<tuple_weights_iterator_type, tuple_particles_iterator_type>::type iterator_tuple;

	typedef typename  detail::tuple_type<N,typename particles_type::const_iterator>::type tuple_particles_const_iterator_type;
	typedef typename  thrust::tuple<typename weights_type::const_iterator> tuple_weights_const_iterator_type;
	typedef typename  detail::tuple_cat_type<tuple_weights_const_iterator_type, tuple_particles_const_iterator_type>::type const_iterator_tuple;


	typedef typename  thrust::tuple< weight_one_iterator> tuple_weight_one_iterator;
	typedef typename  detail::tuple_cat_type<tuple_weight_one_iterator, tuple_particles_iterator_type>::type accpeted_iterator_tuple;


	//reverse iterators
	typedef typename  detail::tuple_type<N,typename particles_type::reverse_iterator>::type tuple_particles_riterator_type;
	typedef typename  thrust::tuple<typename weights_type::reverse_iterator> tuple_weights_riterator_type;
	typedef typename  detail::tuple_cat_type<tuple_weights_riterator_type, tuple_particles_riterator_type>::type reverse_iterator_tuple;

	typedef typename  detail::tuple_type<N,typename particles_type::const_reverse_iterator>::type tuple_particles_const_riterator_type;
	typedef typename  thrust::tuple<typename weights_type::const_reverse_iterator> tuple_weights_const_riterator_type;
	typedef typename  detail::tuple_cat_type<tuple_weights_const_riterator_type, tuple_particles_const_riterator_type>::type const_reverse_iterator_tuple;


public:

	typedef size_t size_type;

	typedef typename detail::tuple_type<N,Vector4R>::type particle_tuple;

	//stl-like typedefs
	//weights
	typedef typename weights_type::iterator                                          weights_iterator;
	typedef typename weights_type::const_iterator                                    weights_const_iterator;
	typedef thrust::reverse_iterator<typename weights_type::reverse_iterator>        weights_reverse_iterator;
	typedef thrust::reverse_iterator<typename weights_type::const_reverse_iterator>  weights_const_reverse_iterator;
	typedef typename thrust::iterator_traits<weights_iterator>::difference_type      weights_difference_type;
	typedef typename thrust::iterator_traits<weights_iterator>::value_type           weights_value_type;
	typedef typename thrust::iterator_traits<weights_iterator>::pointer              weights_pointer;
	typedef typename thrust::iterator_traits<weights_iterator>::reference            weights_reference;
	typedef typename thrust::iterator_traits<weights_iterator>::iterator_category    weights_iterator_category;

	//particles
	typedef typename particles_type::iterator                                          particles_iterator;
	typedef typename particles_type::const_iterator                                    particles_const_iterator;
	typedef thrust::reverse_iterator<particles_iterator>                               particles_reverse_iterator;
	typedef thrust::reverse_iterator<particles_const_iterator>                         particles_const_reverse_iterator;
	typedef typename thrust::iterator_traits<particles_iterator>::difference_type      particles_difference_type;
	typedef typename thrust::iterator_traits<particles_iterator>::value_type           particles_value_type;
	typedef typename thrust::iterator_traits<particles_iterator>::reference            particles_reference;
	typedef typename thrust::iterator_traits<particles_iterator>::iterator_category    particles_iterator_category;

	//this container
	//--------------------------------
	//zipped iterators
	typedef thrust::zip_iterator<accpeted_iterator_tuple> accpeted_iterator;


	//direct
	typedef thrust::zip_iterator<iterator_tuple> iterator;
	typedef thrust::zip_iterator<const_iterator_tuple> const_iterator;

	typedef thrust::zip_iterator<tuple_particles_iterator_type> decays_iterator;
	typedef thrust::zip_iterator<tuple_particles_const_iterator_type> decays_const_iterator;

	//reverse
	typedef thrust::zip_iterator<reverse_iterator_tuple> reverse_iterator;
	typedef thrust::zip_iterator<const_reverse_iterator_tuple> const_reverse_iterator;

	typedef thrust::zip_iterator<tuple_particles_riterator_type> decays_reverse_iterator;
	typedef thrust::zip_iterator<tuple_particles_const_riterator_type> decays_const_reverse_iterator;

	//stl-like typedefs
	typedef typename thrust::iterator_traits<iterator>::difference_type      difference_type;
	typedef typename thrust::iterator_traits<iterator>::value_type           value_type;
	typedef typename thrust::iterator_traits<iterator>::pointer              pointer;
	typedef typename thrust::iterator_traits<iterator>::reference            reference;
	typedef typename thrust::iterator_traits<const_iterator>::reference      const_reference;
	typedef typename thrust::iterator_traits<iterator>::iterator_category    iterator_category;

	typedef typename  detail::tuple_cat_type<weights_pointer_type, particles_pointer_tuple_type>::type pointer_tuple;
	typedef typename  detail::tuple_cat_type<weights_const_pointer_type, particles_const_pointer_tuple_type>::type const_pointer_tuple;



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
	fDecays(other.GetDecays()),
	fWeights(other.GetWeights())
	{
		/*
		fDecays = data_type();
		for( size_t i=0; i<N; i++)
			fDecays[i] = std::move(particles_type(other.begin(i), other.end(i)));
		*/
	}

	/**
	 * Move constructor.
	 * @param other
	 */
	Decays(Decays<N,detail::BackendPolicy<BACKEND>>&& other ):
		fDecays(other.MoveDecays()),
		fWeights(other.MoveWeights())
	{}

	/**
	 * Assignment operator.
	 * @param other
	 */
	template< hydra::detail::Backend BACKEND2>
	Decays(Decays<N,detail::BackendPolicy<BACKEND2>> const& other )
	{
		fWeights = (weights_type(other.wbegin(), other.wend()));

		for( size_t i=0; i<N; i++)
			fDecays[i] = std::move(particles_type(other.pbegin(i), other.pend(i)));

	}

	/**
	 * Assignment operator.
	 * @param other
	 */
	template<typename Iterator>
	Decays( Iterator begin, Iterator end )
	{
		size_t n = thrust::distance(begin, end );
		for( size_t i=0; i<N; i++)	fDecays[i].resize(n);
		fWeights.resize(n);
		thrust::copy(begin, end, this->begin());
	}

	/**
	 * Assignment operator.
	 * @param other
	 */
	Decays<N,detail::BackendPolicy<BACKEND>>&
	operator=(Decays<N,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(*this==&other) return *this;
		this->fDecays  = other.GetDecays();
		this->fWeights = other.GetWeights();

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
		if(*this==&other) return *this;
		this->fDecays  = other.MoveDecays();
		this->fWeights = other.MoveWeights();
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
		if(*this==&other) return *this;
		this->fWeights = std::move(weights_type(other.wbegin(), other.wend()));

		for( size_t i=0; i<N; i++)
			this->fDecays[i] = std::move(particles_type(other.pbegin(i), other.pend(i)));

		return *this;
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is a tuple with N final state particles.
	 */
	void AddDecay(GReal_t w, const particle_tuple& p ){
		this->push_back(w , p);
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is an array with N final state particles.
	 */
	void AddDecay(GReal_t w, Vector4R const (&p)[N]){
		this->push_back(w , p);
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param w is the weight of the decay being added.
	 * @param p is a braced list with N final state particles.
	 */
	void AddDecay(GReal_t w, std::initializer_list<Vector4R>const& p){
		this->push_back(w , p);
	}

	/**
	 * Add a decay to the container, increasing its size by one element.
	 * @param value is a hydra::tuple<double, Vector4R,...,Vector4R >
	 *  = {weight, \f$p_1,...,p_N\f$}.
	 */
	void AddDecay(value_type const& value){
		this->push_back(value);
	}

	/**
	 * Get the range containing the particle number \f$i\f$.
	 * @param i index of particle.
	 * @return std::pair of iterators {begin, end}.
	 */
    hydra::pair<particles_iterator, particles_iterator>
    GetParticles(size_t i){
    	return hydra::make_pair(this->pbegin(i), this->pend(i));
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
    hydra::pair<particles_iterator, particles_iterator>
    GetParticleComponents(size_t i, size_t j){
      	return hydra::make_pair(this->pcbegin(i, j), this->pcend(i, j));
      }

    const typename particles_type::vector_type&
    GetListOfParticleComponents(size_t i, size_t j){
          	return fDecays[i].column(j);
    }

    /**
     * Get a reference a decay.
     * @param i index of the decay.
     * @return reference a decay
     */
    reference GetDecay(size_t i){
    	return this->begin()[i];
    }

    /**
     * Get a constant reference a decay.
     * @param i index of the decay.
     * @return reference a decay
     */
    reference GetDecay(size_t i) const {
    	return this->begin()[i];
    }

    /**
     * Get a range pointing to a set of unweighted events.
     * This method will re-order the container to group together
     * accepted events and return a pair of iterators with ready-only
     * access to container.
     *
     * @return std::pair with iterators pointing to a range of unweigted
     * particles.
     */
    hydra::pair<accpeted_iterator, accpeted_iterator>
    Unweight(GUInt_t scale=1.0);

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
     * auto bw = [ ]__host__ __device__(size_t n, hydra::Vector4R* particles )
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
    hydra::pair<accpeted_iterator, accpeted_iterator>
    Unweight( FUNCTOR  const& functor, GUInt_t scale);

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
     * auto bw = [ ]__host__ __device__(size_t n, hydra::Vector4R* particles )
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
	inline void pop_back();

	inline void push_back(GReal_t weight, const particle_tuple& particles);

	inline void push_back(GReal_t weight, Vector4R const (&particles)[N]);

	inline void	push_back(GReal_t weight, std::initializer_list<Vector4R>const& list_args);

	inline void	push_back(value_type const& value);

	void resize(size_t size);

	void clear();

	void shrink_to_fit();

	void reserve(size_t size);

	size_t size() const;

	size_t capacity() const;

	bool empty() const;

	iterator erase(iterator pos);

	iterator erase(iterator first, iterator last);

	iterator insert(iterator position, const value_type &x);

	void insert(iterator position, size_type n, const value_type &x);


	template<typename InputIterator>
	void insert(iterator position, InputIterator first, InputIterator last);

	reference front();

	const_reference front() const;

	reference back();

	const_reference back() const;

	typename particles_type::vpointer pcdata( size_t particle, size_t component );

	typename particles_type::vpointer pcdata( size_t particle, size_t component ) const;

    weights_pointer wdata( );

    weights_pointer wdata( ) const;

	//non-constant access
	iterator begin();

	iterator end();

	//non-constant access
	reverse_iterator rbegin();

	reverse_iterator rend();

	//constant access
	const_iterator begin() const;

	const_iterator end() const;

	const_reverse_iterator rbegin() const;

	const_reverse_iterator rend() const;

	const_iterator cbegin() const;

	const_iterator cend() const;

	const_reverse_iterator crbegin() const;

	const_reverse_iterator crend() const;


	//non-constant access
	decays_iterator pbegin();

	decays_iterator pend();

	decays_reverse_iterator prbegin();

	decays_reverse_iterator prend();

	particles_iterator pbegin(size_t i);

	particles_iterator pend(size_t i);

	particles_reverse_iterator prbegin(size_t i);

	particles_reverse_iterator prend(size_t i);

	weights_iterator wbegin();

	weights_iterator wend();

	weights_reverse_iterator wrbegin();

	weights_reverse_iterator wrend();

	//constant access const
	decays_const_iterator pbegin() const;

	decays_const_iterator pend() const;

	decays_const_reverse_iterator prbegin() const;

	decays_const_reverse_iterator prend() const;


    particles_const_iterator pbegin(size_t i) const;

    particles_const_iterator pend(size_t i) const;

    particles_const_reverse_iterator prbegin(size_t i) const;

    particles_const_reverse_iterator prend(size_t i) const;

    weights_const_iterator wbegin() const;

    weights_const_iterator wend() const;

    weights_const_reverse_iterator wrbegin() const;

    weights_const_reverse_iterator wrend() const;

	//
	inline	reference operator[](size_t n)
	{	return begin()[n] ;	}

	inline const_reference operator[](size_t n) const
	{	return cbegin()[n]; }



private:



	template<size_t I >
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_insert(size_t pos, tuple_particles_iterator_type& output_iterator,  value_type const& value)
	{}

	template<size_t I = 0>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_insert(size_t pos, tuple_particles_iterator_type& output_iterator,  value_type const& value)
	{
		get<I>(output_iterator) = fDecays[I].insert(fDecays[I].begin()+pos, get<I+1>(value));
	    do_insert<I+1>(pos, output_iterator, value );
	}

	template<size_t I,  template<typename ...> class Tuple, typename ...Iterators >
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_insert(size_t pos, Tuple<Iterators...>  first, Tuple<Iterators...> last )
	{}

	template<size_t I = 0,  template<typename ...> class Tuple, typename ...Iterators >
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_insert(size_t pos, Tuple<Iterators...> first, Tuple<Iterators...> last )
	{
		fDecays[I].insert(fDecays[I].begin()+pos, get<I+1>(first.get_iterator_tuple()),
				get<I+1>(last.get_iterator_tuple()));
		do_insert<I+1>(pos, first, last);
	}

	template<size_t I >
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_push_back(value_type const& value)
	{}

	template<size_t I = 0>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_push_back(value_type const& value)
	{
		fDecays[I].push_back( get<I+1>(value));
		do_push_back<I+1>( value );
	}

	template<size_t I >
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_assignment(accpeted_iterator_tuple& left, iterator_tuple& right  )
	{}

	template<size_t I = 1>
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_assignment(accpeted_iterator_tuple& left, iterator_tuple& right  )
	{
		get<I>(left)= get<I>(right);
		do_assignment<I+1>(left, right );
	}


	 const decays_type& GetDecays() const
	{
		return fDecays;
	}

	 const weights_type& GetWeights() const
	{
		return fWeights;
	}

	decays_type MoveDecays()
	{
		return std::move(fDecays);
	}

	weights_type MoveWeights()
	{
		return std::move(fWeights);
	}

	decays_type  fDecays;
	weights_type fWeights;
};

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
