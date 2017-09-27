/*----------------------------------------------------------------------------
 *
 *   Copyright (C) 2016-2017 Antonio Augusto Alves Junior
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
 * DenseHistogram.h
 *
 *  Created on: 22/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef DENSEHISTOGRAM_H_
#define DENSEHISTOGRAM_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>

#include <type_traits>
#include <utility>
#include <array>

namespace hydra {

namespace detail {

template<size_t N>
struct is_multidimensional:
		std::conditional<(N>1),  std::true_type ,std::false_type>::type {};

}//namespace detail

template<size_t N, typename T, typename  BACKEND,
    typename = typename detail::is_multidimensional<N>::type,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
class DenseHistogram;

template<size_t N, typename T, hydra::detail::Backend  BACKEND>
class DenseHistogram<N, T, hydra::detail::BackendPolicy<BACKEND>, std::true_type >
{

	typedef typename hydra::detail::BackendPolicy<BACKEND> system_t;

	typedef typename system_t::template container<double> storage_t;
	typedef typename system_t::template container<T>::iterator iterator;
	typedef typename system_t::template container<T>::const_iterator const_iterator;
	typedef typename iterator::reference reference;

public:

	DenseHistogram()=delete;


	DenseHistogram( std::array<size_t, N> grid,
			std::array<T, N> lowerlimits,   std::array<T, N> upperlimits):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins *=grid[i];
		}

		fContents.resize(fNBins +2 );
	}

	DenseHistogram( size_t (&grid)[N],
			T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins*=grid[i];
		}

		fContents.resize(fNBins  +2);
	}


	DenseHistogram(DenseHistogram<N, T,hydra::detail::BackendPolicy<BACKEND>> const& other ):
		fContents(other.GetContents())
	{
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
	}


	template< hydra::detail::Backend  BACKEND2>
	DenseHistogram(DenseHistogram<N, T, hydra::detail::BackendPolicy<BACKEND2>> const& other ):
		fContents(other.GetContents())
	{
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
	}



	const storage_t& GetContents() const {
		return fContents;
	}

	void SetContents(storage_t histogram) {
		fContents = histogram;
	}

	size_t GetGrid(size_t i) const {
		return fGrid[i];
	}

	T GetLowerLimits(size_t i) const {
		return fLowerLimits[i];
	}

	T GetUpperLimits(size_t i) const {
		return fUpperLimits[i];
	}

	size_t GetNBins() const {
		return fNBins;
	}
	//stl interface
	iterator begin(){
		return fContents.begin();
	}

	iterator end(){
		return fContents.end();
	}

	const_iterator begin() const {
		return fContents.begin();
	}

	const_iterator end() const {
		return fContents.end();
	}

	reference operator[](size_t i) {
		return *(fContents.begin()+i);
	}

	const reference operator[](size_t i) const {
		return fContents.begin()[i];
	}



	size_t size() const
	{
		return  HYDRA_EXTERNAL_NS::thrust::distance(fContents.begin(), fContents.end() );
	}

	template<typename Iterator>
	void Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	void Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

private:

	T fUpperLimits[N];
	T fLowerLimits[N];
	size_t   fGrid[N];
	size_t   fNBins;
	storage_t fContents;


};

template< typename T, hydra::detail::Backend  BACKEND >
class DenseHistogram<1, T, hydra::detail::BackendPolicy<BACKEND>, std::false_type >{

	typedef typename hydra::detail::BackendPolicy<BACKEND> system_t;

	typedef typename system_t::template container<double> storage_t;

	typedef typename storage_t::iterator iterator;
	typedef typename storage_t::const_iterator const_iterator;

	typedef typename iterator::reference reference;

public:

	DenseHistogram()=delete;


	DenseHistogram( size_t grid, T lowerlimits, T upperlimits):
		fGrid(grid),
		fLowerLimits(lowerlimits),
		fUpperLimits(upperlimits),
		fNBins(grid),
		fContents( grid+2 )
	{}


	DenseHistogram(DenseHistogram<1, T,hydra::detail::BackendPolicy<BACKEND>> const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}

	template< hydra::detail::Backend  BACKEND2>
	DenseHistogram(DenseHistogram<1, T, hydra::detail::BackendPolicy<BACKEND2>> const& other ):
	fContents(other.GetContents()),
	fGrid(other.GetGrid()),
	fLowerLimits(other.GetLowerLimits()),
	fUpperLimits(other.GetUpperLimits()),
	fNBins(other.GetNBins())
	{}



	const storage_t& GetContents()const  {
		return fContents;
	}

	void SetContents(storage_t histogram) {
		fContents = histogram;
	}

	size_t GetGrid() const {
		return fGrid;
	}

	T GetLowerLimits() const {
		return fLowerLimits;
	}

	T GetUpperLimits() const {
		return fUpperLimits;
	}

	size_t GetNBins() const {
		return fNBins;
	}

	//stl interface
	iterator begin(){
		return fContents.begin();
	}

	iterator end(){
		return fContents.end();
	}

	const_iterator begin() const {
		return fContents.begin();
	}

	const_iterator end() const {
		return fContents.end();
	}

   reference operator[](size_t i) {
    	return *(fContents.begin()+i);
    }

    const reference operator[](size_t i) const {
        return fContents.begin()[i];
    }



	size_t size() const
	{
	 return  HYDRA_EXTERNAL_NS::thrust::distance(fContents.begin(), fContents.end() );
	}

	template<typename Iterator>
	void Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	void Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

private:

	T fUpperLimits;
	T fLowerLimits;
	size_t   fGrid;
	size_t   fNBins;
	storage_t fContents;


};


}  // namespace hydra

#include <hydra/detail/DenseHistogram.inl>


#endif /* DENSEHISTOGRAM_H_ */
