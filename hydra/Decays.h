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
//thrust
#include <thrust/copy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/reverse_iterator.h>

namespace hydra {

template<size_t N, typename BACKEND>
class Decays;

template<size_t N, hydra::detail::Backend BACKEND>
class Decays<N, hydra::detail::BackendPolicy<BACKEND> > {

	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	typedef multiarray<4,GReal_t, hydra::detail::BackendPolicy<BACKEND> > particles_type;
	typedef std::array<particles_type, N> decays_type;
	typedef typename system_t::template container<GReal_t> weights_type;

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
	typedef thrust::reverse_iterator<typename particles_type::reverse_iterator>        particles_reverse_iterator;
	typedef thrust::reverse_iterator<typename particles_type::const_reverse_iterator>  particles_const_reverse_iterator;
	typedef typename thrust::iterator_traits<particles_iterator>::difference_type      particles_difference_type;
	typedef typename thrust::iterator_traits<particles_iterator>::value_type           particles_value_type;
	typedef typename thrust::iterator_traits<particles_iterator>::pointer              particles_pointer;
	typedef typename thrust::iterator_traits<particles_iterator>::reference            particles_reference;
	typedef typename thrust::iterator_traits<particles_iterator>::iterator_category    particles_iterator_category;

	//decays
	typedef typename decays_type::iterator                                          decays_iterator;
	typedef typename decays_type::const_iterator                                    decays_const_iterator;
	typedef thrust::reverse_iterator<typename decays_type::reverse_iterator>        decays_reverse_iterator;
	typedef thrust::reverse_iterator<typename decays_type::const_reverse_iterator>  decays_const_reverse_iterator;
	typedef typename thrust::iterator_traits<decays_iterator>::difference_type      decays_difference_type;
	typedef typename thrust::iterator_traits<decays_iterator>::value_type           decays_value_type;
	typedef typename thrust::iterator_traits<decays_iterator>::pointer              decays_pointer;
	typedef typename thrust::iterator_traits<decays_iterator>::reference            decays_reference;
	typedef typename thrust::iterator_traits<decays_iterator>::iterator_category    decays_iterator_category;

	//this container
	//--------------------------------
	//zipped iterators
	//direct
	typedef thrust::zip_iterator<iterator_tuple> iterator;
	typedef thrust::zip_iterator<const_iterator_tuple> const_iterator;
	//reverse
	typedef thrust::zip_iterator<reverse_iterator_tuple> reverse_iterator;
	typedef thrust::zip_iterator<const_reverse_iterator_tuple> const_reverse_iterator;

	//stl-like typedefs
	typedef typename thrust::iterator_traits<iterator>::difference_type      difference_type;
	typedef typename thrust::iterator_traits<iterator>::value_type           value_type;
	typedef typename thrust::iterator_traits<iterator>::pointer              pointer;
	typedef typename thrust::iterator_traits<iterator>::reference            reference;
	typedef typename thrust::iterator_traits<const_iterator>::reference      const_reference;
	typedef typename thrust::iterator_traits<iterator>::iterator_category    iterator_category;

	typedef typename  detail::tuple_cat_type<weights_pointer_type, particles_pointer_tuple_type>::type pointer_tuple;
	typedef typename  detail::tuple_cat_type<weights_const_pointer_type, particles_const_pointer_tuple_type>::type const_pointer_tuple;


	Decays():
		fDecays(decays_type()),
		fWeights(weights_type())
	{};

	Decays(size_t n):
	fDecays(decays_type()),
	fWeights(weights_type(n))
	{
		for( size_t i=0; i<N; i++)
			fDecays[i].resize(n);
	};

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

	Decays(Decays<N,detail::BackendPolicy<BACKEND>>&& other ):
		fDecays(other.MoveDecays()),
		fWeights(other.MoveWeights())
	{}

	template< hydra::detail::Backend BACKEND2>
	Decays(Decays<N,detail::BackendPolicy<BACKEND2>> const& other ):
	fDecays(other.GetDecays()),
	fWeights(other.GetWeights())
	{
		/*
		fDecays = data_type();
		for( size_t i=0; i<N; i++)
			fDecays[i] = std::move( particles_type( other.begin(i), other.end(i) ) );
			*/
	}

	Decays<N,detail::BackendPolicy<BACKEND>>&
	operator=(Decays<N,detail::BackendPolicy<BACKEND>> const& other )
	{
		if(*this==&other) return *this;
		this->fDecays  = other.GetDecays();
		this->fWeights = other.GetWeights();
		/*
		for( size_t i=0; i<N; i++)
			this->fDecays[i] = std::move(particles_type(other.begin(), other.end()));
		 */
		return *this;
	}

	Decays<N,detail::BackendPolicy<BACKEND>>&
	operator=(Decays<N,detail::BackendPolicy<BACKEND> >&& other )
	{
		if(*this==&other) return *this;
		this->fDecays  = other.MoveDecays();
		this->fWeights = other.MoveWeights();
		return *this;
	}

	template< hydra::detail::Backend BACKEND2>
	Decays<N,detail::BackendPolicy<BACKEND> >&
	operator=(Decays<N,detail::BackendPolicy<BACKEND2> > const& other )
	{
		if(*this==&other) return *this;
		this->fDecays  = other.GetDecays();
		this->fWeights = other.GetWeights();
		/*
		for( size_t i=0; i<N; i++)
			this->fDecays[i] = std::move( vector_t( other.begin(i), other.end(i) ) );
			*/
		return *this;
	}

	//stl compliant interface
	//-----------------------

	/**
	 * @
	 */
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

	particles_pointer pdata( size_t particle, size_t component );

	particles_pointer pdata( size_t particle, size_t component ) const;

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
	particles_iterator pbegin(size_t i);

	particles_iterator pend(size_t i);

	particles_reverse_iterator prbegin(size_t i);

	particles_reverse_iterator prend(size_t i);

	weights_iterator wbegin();

	weights_iterator wend();

	weights_reverse_iterator wrbegin();

	weights_reverse_iterator wrend();

	//constant access const
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

	template<size_t I, typename InputIterator >
	inline typename thrust::detail::enable_if<(I == N), void >::type
	do_insert(size_t pos, InputIterator first, InputIterator last )
	{}

	template<size_t I = 0, typename InputIterator >
	inline typename thrust::detail::enable_if<(I < N), void >::type
	do_insert(size_t pos, InputIterator first, InputIterator last )
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
