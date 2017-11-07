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
#include <hydra/detail/Dimensionality.h>

#include <type_traits>
#include <utility>
#include <array>
#include <vector>

namespace hydra {


template<size_t N, typename T, typename = typename detail::dimensionality<N>::type,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
class DenseHistogram;

template<size_t N, typename T>
class DenseHistogram<N, T, detail::multidimensional>
{


	typedef std::vector<double> storage_t;
	typedef typename storage_t::iterator iterator;
	typedef typename storage_t::const_iterator const_iterator;
	typedef typename storage_t::reference reference;
	typedef typename storage_t::pointer pointer;
	typedef typename storage_t::value_type value_type;

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

	DenseHistogram<N, T, detail::multidimensional>&
	operator=(DenseHistogram<N, T, detail::multidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}

	DenseHistogram(DenseHistogram<N, T, detail::multidimensional> const& other ):
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

	size_t GetBin( size_t  (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	size_t GetBin( std::array<size_t,N> const&  bins){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return bin;
	}

	void GetIndexes(size_t globalbin,  size_t  (&bins)[N]){

		get_indexes(globalbin, bins);
	}

	void GetIndexes(size_t globalbin, std::array<size_t,N>&  bins){

		get_indexes(globalbin, bins);
	}

	double GetBinContent( size_t  (&bins)[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		return (bin >=0 ) && ( bin < (fNBins) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}

	double GetBinContent( size_t  bin){

		return (bin >=0 ) && ( bin<= (fNBins+1) ) ?
				fContents.begin()[bin] :
				std::numeric_limits<double>::max();
	}


	//stl range interface

	pointer data(){
		return fContents.data();
	}

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

    value_type operator[](size_t i) const {
		return fContents.begin()[i];
	}

	size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(fContents.begin(), fContents.end() );
	}

	template<typename Iterator>
	void Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	void Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

private:

	//k = i_1*(dim_2*...*dim_n) + i_2*(dim_3*...*dim_n) + ... + i_{n-1}*dim_n + i_n

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< I== N, void>::type
	get_global_bin( size_t (&indexes)[N], size_t& index){ }

	template<size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N), void>::type
	get_global_bin( size_t (&indexes)[N], size_t& index)
	{
		size_t prod =1;

		for(size_t i=N-1; i>I; i--)
			prod *=fGrid[i];

		index += prod*indexes[I];

		get_global_bin<I+1>( indexes, index);
	}

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< I== N, void>::type
	get_global_bin( std::array<size_t,N> const& indexes, size_t& index){ }

	template<size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N), void>::type
	get_global_bin( std::array<size_t,N> const& indexes, size_t& index)
	{
		size_t prod =1;

		for(size_t i=N-1; i>I; i--)
			prod *=fGrid[i];

		index += prod*indexes[I];

		get_global_bin<I+1>( indexes, index);
	}

	/*
	 *  conversion of one-dimensional index to multidimensional one
	 * ____________________________________________________________
	 */

	//----------------------------------------
	// multiply  std::array elements
	//----------------------------------------
	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply( std::array<size_t, N> const&  obj, size_t& result )
	{ }

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply( std::array<size_t, N> const&  obj, size_t& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply<I+1>( obj, result );
	}

	//----------------------------------------
	// multiply static array elements
	//----------------------------------------
	template< size_t I>
	typename std::enable_if< (I==N), void  >::type
	multiply( size_t (&obj)[N] , size_t& result )
	{ }

	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	multiply( size_t (&obj)[N], size_t& result )
	{
		result = I==0? 1.0: result;
		result *= obj[I];
		multiply<I+1>( obj, result );
	}


	//-------------------------
	// std::array version
	//-------------------------
	//end of recursion
	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t index,  std::array<size_t,N>& indexes)
	{}

	//begin of the recursion
	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index, std::array<size_t,N>& indexes)
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes<I+1>(next_index,indexes );
	}

	//-------------------------
	// static array version
	//-------------------------
	//end of recursion
	template<size_t I>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t index,  size_t (&indexes)[N])
	{}

	//begin of the recursion
	template<size_t I=0>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index,  size_t (&indexes)[N] )
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< I+1>(next_index, indexes );
	}



	T fUpperLimits[N];
	T fLowerLimits[N];
	size_t   fGrid[N];
	size_t   fNBins;
	storage_t fContents;


};


/*
 * 1D dimension specialization
 */
template< typename T >
class DenseHistogram<1, T,  detail::unidimensional >{

	typedef std::vector<double> storage_t;
	typedef typename storage_t::iterator iterator;
	typedef typename storage_t::const_iterator const_iterator;
	typedef typename storage_t::reference reference;
	typedef typename storage_t::pointer pointer;
	typedef typename storage_t::value_type value_type;

public:

	DenseHistogram()=delete;


	DenseHistogram( size_t grid, T lowerlimits, T upperlimits):
		fGrid(grid),
		fLowerLimits(lowerlimits),
		fUpperLimits(upperlimits),
		fNBins(grid),
		fContents( grid+2 )
	{}


	DenseHistogram(DenseHistogram<1, T,detail::unidimensional > const& other ):
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

	double GetBinContent(size_t i){

		return (i>=0) && (i<=fNBins+1) ?
				fContents.begin()[i] :
					std::numeric_limits<double>::max();
	}

	//stl interface
	pointer data(){
		return fContents.data();
	}

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

   value_type  operator[](size_t i) const {
        return fContents.begin()[i];
    }

	size_t size() const	{
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
