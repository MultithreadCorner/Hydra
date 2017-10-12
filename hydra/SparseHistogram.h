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
 * SparseHistogram.h
 *
 *  Created on: 29/09/2017
 *      Author: Antonio Augusto Alves Junior
 */

#ifndef SPARSEHISTOGRAM_H_
#define SPARSEHISTOGRAM_H_

#include <hydra/detail/Config.h>
#include <hydra/detail/BackendPolicy.h>
#include <hydra/Types.h>
#include <hydra/detail/Dimensionality.h>


#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <tuple>
#include <algorithm>

#include <hydra/detail/external/thrust/iterator/iterator_traits.h>
#include <hydra/detail/external/thrust/iterator/zip_iterator.h>

namespace hydra {

template<size_t N, typename T, typename = typename detail::dimensionality<N>::type,
	typename = typename std::enable_if<std::is_arithmetic<T>::value, void>::type>
class SparseHistogram;

template<size_t N, typename T>
class SparseHistogram<N, T, detail::multidimensional>
{

	/// @todo Update to use multivector
	typedef std::vector<double> storage_data_t;
	typedef std::vector<size_t> storage_keys_t;

	typedef typename storage_data_t::iterator data_iterator;
	typedef typename storage_data_t::const_iterator data_const_iterator;
	typedef typename storage_data_t::reference data_reference;
	typedef typename storage_data_t::pointer data_pointer;
	typedef typename storage_data_t::value_type data_value_type;

	typedef typename storage_keys_t::iterator keys_iterator;
	typedef typename storage_keys_t::const_iterator keys_const_iterator;
	typedef typename storage_keys_t::reference keys_reference;
	typedef typename storage_keys_t::pointer keys_pointer;
	typedef typename storage_keys_t::value_type keys_value_type;

	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_iterator, data_iterator>> iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_const_iterator, data_const_iterator> > const_iterator;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	typedef std::pair<size_t*, double*> pointer_pair;

public:

	SparseHistogram()=delete;

	template<typename Int,
		typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	SparseHistogram( std::array<Int, N> grid,
			std::array<T, N> lowerlimits,   std::array<T, N> upperlimits):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins *=grid[i];
		}

	}

	SparseHistogram( size_t (&grid)[N],
			T (&lowerlimits)[N],   T (&upperlimits)[N] ):
				fNBins(1)
	{
		for( size_t i=0; i<N; i++){
			fGrid[i]=grid[i];
			fLowerLimits[i]=lowerlimits[i];
			fUpperLimits[i]=upperlimits[i];
			fNBins*=grid[i];
		}

	}

	SparseHistogram<N, T, detail::multidimensional>&
	operator=(SparseHistogram<N, T, detail::multidimensional> const& other )
	{
		if(this==&other) return *this;

		fContents = other.GetContents();
		fBins = other.GetBins();

		for( size_t i=0; i<N; i++){
			fGrid[i] = other.GetGrid(i);
			fLowerLimits[i] = other.GetLowerLimits(i);
			fUpperLimits[i] = other.GetUpperLimits(i);
		}

		fNBins= other.GetNBins();
		return *this;
	}

	SparseHistogram(SparseHistogram<N, T, detail::multidimensional> const& other ):
			fContents(other.GetContents()),
			fBins(other.GetBins())
		{
			for( size_t i=0; i<N; i++){
				fGrid[i] = other.GetGrid(i);
				fLowerLimits[i] = other.GetLowerLimits(i);
				fUpperLimits[i] = other.GetUpperLimits(i);
			}

			fNBins= other.GetNBins();
		}

	const storage_data_t& GetContents() const {
		return fContents;
	}

	const storage_keys_t& GetBins() const
	{
		return fBins;
	}

	void SetBins(storage_keys_t bins)
	{
		fBins = bins;
	}

	void SetContents(storage_data_t histogram) {
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

	template<typename Int,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	void GetIndexes(size_t globalbin,  Int  (&bins)[N]){

		get_indexes(globalbin, bins);
	}

	template<typename Int,
				typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	void GetIndexes(size_t globalbin, std::array<Int,N>&  bins){

		get_indexes(globalbin, bins);
	}

	double GetBinContent( size_t  bins[N]){

		size_t bin=0;

		get_global_bin( bins,  bin);

		size_t index = std::distance(fBins.begin(),
				std::find(fBins.begin(),fBins.end(), bin));

		return  ( bin< fBins.size() ) ?
				fContents.begin()[bin] : 0.0;
	}

	double GetBinContent( size_t  bin){

		size_t index = std::distance(fBins.begin(),
				std::find(fBins.begin(),fBins.end(), bin));

		return  ( bin< fBins.size() ) ?
				fContents.begin()[bin] : 0.0;
	}


	//stl interface

	pointer_pair data(){
		return std::make_pair(fBins.data() , fContents.data());
	}

	iterator begin(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
						HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.begin() ,  fContents.begin()) );
	}

	iterator end(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	const_iterator begin() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.cbegin() ,  fContents.cbegin() ) );
	}

	const_iterator end() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	reference operator[](size_t i) {
		return *(begin()+i);
	}

    value_type operator[](size_t i) const {
		return begin()[i];
	}

	size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(begin(), end() );
	}

	template<typename Iterator>
	void Fill(Iterator begin, Iterator end);

	template<typename Iterator1, typename Iterator2>
	void Fill(Iterator1 begin, Iterator1 end, Iterator2 wbegin);

private:

	//k = i_1*(dim_2*...*dim_n) + i_2*(dim_3*...*dim_n) + ... + i_{n-1}*dim_n + i_n

	template<size_t I>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< I== N, void>::type
	get_global_bin(const size_t (&indexes)[N], size_t& index){ }

	template<size_t I=0>
	typename HYDRA_EXTERNAL_NS::thrust::detail::enable_if< (I< N), void>::type
	get_global_bin(const size_t (&indexes)[N], size_t& index)
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
	template<typename Int, size_t I,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t index,  std::array<Int,N>& indexes)
	{}

	//begin of the recursion
	template<typename Int, size_t I=0,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index, std::array<Int,N>& indexes)
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< Int,I+1>(next_index,indexes );
	}

	//-------------------------
	// static array version
	//-------------------------
	//end of recursion
	template<typename Int, size_t I,
	typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I==N), void  >::type
	get_indexes(size_t index, Int (&indexes)[N])
	{}

	//begin of the recursion
	template<typename Int, size_t I=0,
			typename = typename std::enable_if<std::is_integral<Int>::value, void>::type>
	typename std::enable_if< (I<N), void  >::type
	get_indexes(size_t index, Int (&indexes)[N] )
	{
		size_t factor    =  1;
		multiply<I+1>(fGrid, factor );
		indexes[I]  =  index/factor;
		size_t next_index =  index%factor;
		get_indexes< Int, I+1>(next_index, indexes );
	}


	T fUpperLimits[N];
	T fLowerLimits[N];
	size_t   fGrid[N];
	size_t   fNBins;
	storage_data_t fContents;
	storage_keys_t fBins;


};


/*
 * 1D dimension specialization
 */
template< typename T >
class SparseHistogram<1, T,  detail::unidimensional >{


	/// @todo Update to use multivector
	typedef std::vector<double> storage_data_t;
	typedef std::vector<size_t> storage_keys_t;

	typedef typename storage_data_t::iterator data_iterator;
	typedef typename storage_data_t::const_iterator data_const_iterator;
	typedef typename storage_data_t::reference data_reference;
	typedef typename storage_data_t::pointer data_pointer;
	typedef typename storage_data_t::value_type data_value_type;

	typedef typename storage_keys_t::iterator keys_iterator;
	typedef typename storage_keys_t::const_iterator keys_const_iterator;
	typedef typename storage_keys_t::reference keys_reference;
	typedef typename storage_keys_t::pointer keys_pointer;
	typedef typename storage_keys_t::value_type keys_value_type;


	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_iterator, data_iterator>> iterator;
	typedef HYDRA_EXTERNAL_NS::thrust::zip_iterator<
			HYDRA_EXTERNAL_NS::thrust::tuple<keys_const_iterator, data_const_iterator> > const_iterator;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::reference reference;
	typedef typename HYDRA_EXTERNAL_NS::thrust::iterator_traits<iterator>::value_type value_type;
	typedef std::pair<size_t*, double*> pointer_pair;

public:

	SparseHistogram()=delete;


	SparseHistogram( size_t grid, T lowerlimits, T upperlimits):
		fGrid(grid),
		fLowerLimits(lowerlimits),
		fUpperLimits(upperlimits),
		fNBins(grid)
	{}


	SparseHistogram(SparseHistogram<1, T,detail::unidimensional > const& other ):
		fContents(other.GetContents()),
		fGrid(other.GetGrid()),
		fLowerLimits(other.GetLowerLimits()),
		fUpperLimits(other.GetUpperLimits()),
		fNBins(other.GetNBins())
	{}


	const storage_data_t& GetContents()const  {
		return fContents;
	}

	void SetContents(storage_data_t histogram) {
		fContents = histogram;
	}

	const storage_keys_t& GetBins() const
	{
		return fBins;
	}

	void SetBins(storage_keys_t bins)
	{
		fBins = bins;
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


	double GetBinContent( size_t  bin){

		size_t index = std::distance(fBins.begin(),
				std::find(fBins.begin(),fBins.end(), bin));

		return  ( bin< fBins.size() ) ?
				fContents.begin()[bin] : 0.0;
	}


	//stl interface

	pointer_pair data(){
		return std::make_pair(fBins.data() , fContents.data());
	}

	iterator begin(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
						HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.begin() ,  fContents.begin()) );
	}

	iterator end(){
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	const_iterator begin() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.cbegin() ,  fContents.cbegin() ) );
	}

	const_iterator end() const {
		return HYDRA_EXTERNAL_NS::thrust::make_zip_iterator(
				HYDRA_EXTERNAL_NS::thrust::make_tuple(fBins.end() ,  fContents.end() ));
	}

	reference operator[](size_t i) {
		return *(begin()+i);
	}

    value_type operator[](size_t i) const {
		return begin()[i];
	}

	size_t size() const	{

		return  HYDRA_EXTERNAL_NS::thrust::distance(begin(), end() );
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
	storage_data_t fContents;
	storage_keys_t fBins;

};



}  // namespace hydra

#include <hydra/detail/SparseHistogram.inl>

#endif /* SPARSEHISTOGRAM_H_ */
