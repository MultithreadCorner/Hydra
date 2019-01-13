/*----------------------------------------------------------------------------
 *
+9 *   Copyright (C) 2016 - 2019 Antonio Augusto Alves Junior
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
 * Vegas.inl
 *
 *  Created on: 19/07/2016
 *      Author: Antonio Augusto Alves Junior
 */


//#ifndef VEGAS_INL_
//#define VEGAS_INL_

#include <hydra/detail/Config.h>
#include <hydra/Containers.h>
#include <hydra/Types.h>
#include <hydra/VegasState.h>
#include <hydra/detail/utility/StreamSTL.h>
#include <hydra/detail/functors/ProcessCallsVegas.h>

//std
#include <chrono>
#include <iostream>
#include <cstdio>
#include <utility>

//thrust
#include <hydra/detail/external/thrust/transform_reduce.h>
#include <hydra/detail/external/thrust/sort.h>


#define USE_ORIGINAL_CHISQ_FORMULA 0
#define CALLS_PER_BOX 2.0

namespace hydra {

template<size_t N, hydra::detail::Backend  BACKEND, typename GRND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::Integrate(FUNCTOR const& fFunctor )
{

	fState.SetStage(0);

	auto temp = IntegIterator(fFunctor, 1 );

	return IntegIterator(fFunctor, 0 );

}

template<size_t N, hydra::detail::Backend  BACKEND , typename GRND>
template<typename FUNCTOR>
std::pair<GReal_t, GReal_t>
Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::IntegIterator(FUNCTOR const& fFunctor, GBool_t training )
{

	//fState.SetStage(0);

	GReal_t cum_int, cum_sig;


	if( fState.GetStage() == 0) {
		InitGrid();
		fState.ClearStoredIterations();

		if (fState.GetVerbose() >= 0) {
			PrintLimits();
		}
	}

	if( fState.GetStage() <= 1) {

		fState.SetWeightedIntSum(0.0);
		fState.SetSumOfWeights(0.0);
		fState.SetChiSum(0.0);
		fState.SetItNum(1);
		fState.SetSamples(0.0);
		fState.SetChiSquare(0.0);
	}

	if( fState.GetStage()  <= 2) {

		size_t bins = fState.GetNBinsMax();
		size_t boxes = 1;

		if (fState.GetMode() != MODE_IMPORTANCE_ONLY) {
			/* shooting for 2 calls/box */

			boxes = floor( pow(fState.GetCalls(training )/2.0, 1.0 / N ));
		//	if(boxes==1) boxes++;
		//	std::cout << "boxes  " << boxes << " bins " <<fState.GetNBinsMax()<< std::endl;

			fState.SetMode(MODE_IMPORTANCE);

		}


		/*size_t*/GReal_t tot_boxes = pow( (GReal_t)boxes,  N);
		fState.SetCallsPerBox(std::max(  GInt_t(fState.GetCalls(training ) / tot_boxes), 2) );
		fState.SetCalls( training , fState.GetCallsPerBox() * tot_boxes);
		//std::cout << "fState.GetCalls "<< fState.GetCalls()<< std::endl;

		/* total volume of x-space/(avg num of calls/bin) */
		fState.SetJacobian( fState.GetVolume() * pow((GReal_t) bins, (GReal_t)N)/ fState.GetCalls(training) );

		//std::cout << "fState.GetVolume() " << fState.GetVolume() << std::endl;

		fState.SetNBoxes(boxes);

		/* If the number of bins changes from the previous invocation, bins
		 are expanded or contracted accordingly, while preserving bin
		 density */

		if (bins != fState.GetNBins()) {
			ResizeGrid( bins);

			if (fState.GetVerbose() > 1) {
				PrintGrid();
			}
		}

		if (fState.GetVerbose() >= 0) {
			PrintHead();
		}
	}

	fState.SetItStart( fState.GetItNum());

	cum_int = 0.0;
	cum_sig = 0.0;
	size_t nkeys  = N*fState.GetCalls(training);
	fFValInput.resize(nkeys);
	fGlobalBinInput.resize(nkeys);
	fFValOutput.resize(nkeys);
	fGlobalBinOutput.resize(nkeys);

	//for (size_t it = 0; it < fState.GetIterations()+fState.GetTrainingIterations(); it++)

	for (size_t it = 0; it < (!training?fState.GetIterations():fState.GetTrainingIterations()); it++)
	{

		auto start = std::chrono::high_resolution_clock::now();

		GReal_t intgrl = 0.0;
		GReal_t intgrl_sq = 0.0;
		GReal_t tss = 0.0;
		GReal_t wgt=0;
		GReal_t var=0;
		GReal_t sig=0;

		size_t calls_per_box = fState.GetCallsPerBox();
		GReal_t jacbin = fState.GetJacobian();

		//if(it >=fState.GetTrainingIterations())	fState.SetItNum(fState.GetItStart() + it);
		if(!training)	fState.SetItNum(fState.GetItStart() + it);

		ResetGridValues();

		/*
		 * **********************************************
		 * call  accelerator                         *
		 * **********************************************
		 */
		auto start_fc = std::chrono::high_resolution_clock::now();
		ProcessFuncionCalls( fFunctor,training, intgrl,  tss);
		auto end_fc = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_fc = end_fc - start_fc;
		/*
		 * Compute final results for this iteration
		 */
		//if(fState.GetItNum() >= fState.GetTrainingIterations())
		if(!training)
		{

			var = tss / (calls_per_box - 1.0);


			if (var > 0) {
				wgt = 1.0 / var;
			} else if (fState.GetSumOfWeights() > 0) {
				wgt = fState.GetSumOfWeights()/ fState.GetSamples();
			} else {
				wgt = 0.0;
			}

			intgrl_sq = intgrl * intgrl;

			sig = sqrt(var);


			if (wgt > 0.0) {

				GReal_t sum_wgts = fState.GetSumOfWeights();
				GReal_t wtd_int_sum = fState.GetWeightedIntSum();
				GReal_t m = (sum_wgts > 0.0) ? (wtd_int_sum / sum_wgts) : 0.0;
				GReal_t q = intgrl - m;

				fState.SetSamples(fState.GetSamples()+1);
				fState.SetSumOfWeights(fState.GetSumOfWeights() + wgt);
				fState.SetWeightedIntSum(fState.GetWeightedIntSum() + intgrl * wgt);
				fState.SetChiSum(fState.GetChiSum()  + intgrl_sq * wgt );

				cum_int = fState.GetWeightedIntSum() / fState.GetSumOfWeights();
				cum_sig = sqrt(1.0/fState.GetSumOfWeights() );

#if USE_ORIGINAL_CHISQ_FORMULA

				/*
				 * This is the chisq formula from the original Lepage paper.  It
				 * computes the variance from <x^2> - <x>^2 and can suffer from
				 * catastrophic cancellations, e.g. returning negative chisq.
				 */

				if (fState.GetSamples() > 1)
				{
					fState.SetChiSquare((fState.GetChiSum() - fState.GetWeightedIntSum() * cum_int)/(fState.GetSamples() - 1.0));
				}
#else
				/*
				 * The new formula below computes exactly the same quantity as above
				 * but using a stable recurrence
				 */

				if (fState.GetSamples() == 1) {
					fState.SetChiSquare(0.0);
				} else {

					GReal_t chi2 = fState.GetChiSquare();

					chi2 *= (fState.GetSamples() - 2.0);
					chi2 += (wgt / (1 + (wgt / sum_wgts))) * q * q;
					chi2 /= (fState.GetSamples() - 1.0);

					fState.SetChiSquare(chi2);

				}
#endif
			} else {
				cum_int += (intgrl - cum_int) / (it + 1.0);
				cum_sig = 0.0;
			}


			if (fState.GetVerbose() >= 0) {
				PrintResults( intgrl, sig, cum_int, cum_sig, elapsed_fc.count());

				if (it + 1 == fState.GetIterations() && fState.GetVerbose() > 0) {
					PrintGrid();
				}
			}


		}

		if (fState.GetVerbose() > 1) {
			PrintDistribution();
		}

		if(!training && !fState.IsTrainedGridFrozen() )
		 RefineGrid();

		if (fState.GetVerbose() > 1) {
			PrintGrid();
		}

		auto end = std::chrono::high_resolution_clock::now();
		//auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::chrono::duration<double, std::milli> elapsed = end - start;


		fState.SetResult(intgrl);
		fState.SetSigma(sig);
		if(!training)
		{
			fState.StoreIterationResult(intgrl, sig);
			fState.StoreCumulatedResult(cum_int, cum_sig);
			fState.StoreIterationDuration( elapsed.count() );
			fState.StoreFunctionCallsDuration( elapsed_fc.count() );
		}
		//if(it >=fState.GetTrainingIterations())
		if(!training && it > 1)
		{

			if(    (fState.IsUseRelativeError())  && (cum_sig/cum_int < fState.GetMaxError()) ) break;
			if( it > 1 &&  (!fState.IsUseRelativeError()) && (fabs(cum_int - intgrl )/cum_int< fState.GetMaxError()) ) break;
		}
	}

	/* By setting stage to 1 further calls will generate independent
	 estimates based on the same grid, although it may be rebinned. */

	fState.SetStage(1);

	fFValInput=rvector_backend();
	fGlobalBinInput=uvector_backend();
	fFValOutput=rvector_backend();
	fGlobalBinOutput=uvector_backend();

	return std::make_pair(cum_int, cum_sig);


}

template< size_t N , hydra::detail::Backend  BACKEND, typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::PrintLimits()  {

	//PrintToStream(fState.GetOStream(),, ... );

	PrintToStream(fState.GetOStream(), "The limits of Integration are:\n" );
	for (size_t j = 0; j < N; ++j)
		PrintToStream(fState.GetOStream(), "\nxl[%lu]=%f    xu[%lu]=%f", j, fState.GetXLow()[j], j, fState.GetXUp()[j] );

	//fState.GetOStream() << boost::format("The limits of Int_tegration are:\n");
	//for (size_t j = 0; j < N; ++j)
	//	fState.GetOStream() <<  boost::format("\nxl[%lu]=%f    xu[%lu]=%f") % j % fState.GetXLow()[j] % j % fState.GetXUp()[j] ;

	fState.GetOStream() << std::endl;


}

template< size_t N , hydra::detail::Backend  BACKEND, typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::PrintHead()  {

	PrintToStream(fState.GetOStream(),
			"\nnum_dim=%lu, calls=%lu, it_num=%d, max_it_num=%d ",
			N, fState.GetCalls(), fState.GetItNum(), fState.GetIterations() );


	PrintToStream(fState.GetOStream(),
			"verb=%d, alph=%.2f,\nmode=%d, bins=%d, boxes=%d\n" ,
			fState.GetVerbose(), fState.GetAlpha(), fState.GetMode(), fState.GetNBins(), fState.GetNBoxes() );

	PrintToStream(fState.GetOStream(),
			"\n            |-------  single iteration  -------|       |------  accumulated results  ------|  \n");

	PrintToStream(fState.GetOStream(),
			"iteration          Integral          Sigma                    Integral        Sigma            chi-sq/it     duration (ms)/it\n\n");

			//fState.GetOStream() << boost::format("\nnum_dim=%lu, calls=%lu, it_num=%d, max_it_num=%d ") % N
		//		% fState.GetCalls() % fState.GetItNum() % fState.GetIterations() << std::endl;

	//fState.GetOStream() <<  boost::format("verb=%d, alph=%.2f,\nmode=%d, bins=%d, boxes=%d\n")
	//		% fState.fVerbose % fState.GetAlpha() % fState.GetMode()
	//		% fState.GetNBins() % fState.GetNBoxes() << std::endl;

	//fState.GetOStream() << boost::format("\n            |-------  single iteration  -------|       |------  accumulated results  ------|  \n")<< std::endl;

	//fState.GetOStream() << boost::format("iteration          Integral          Sigma                    Integral        Sigma            chi-sq/it     duration (ms)/it\n\n") << std::endl;


}

template< size_t N, hydra::detail::Backend  BACKEND , typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::PrintResults(GReal_t integral, GReal_t sigma,
		GReal_t cumulated_integral, GReal_t cumulated_sigma, GReal_t time)  {

	PrintToStream(fState.GetOStream(),
			"%4d           %6.4e          %10.4e              %6.4e           %10.4e           %10.4e        %10.4e ms\n",
			fState.GetItNum(), integral, sigma, cumulated_integral, cumulated_sigma, fState.GetChiSquare(), time);

	/*
	fState.GetOStream() << boost::format( "%4d           %6.4e          %10.4e              %6.4e           %10.4e           %10.4e        %10.4e ms\n")
			% fState.GetItNum() % integral % sigma % cumulated_integral
			% cumulated_sigma % fState.GetChiSquare() % time ;
			*/


}

template< size_t N, hydra::detail::Backend  BACKEND, typename GRND >
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::PrintDistribution()   {

	size_t i, j;

	/*
	if (!fState.GetVerbose())
		return;*/

	for (j = 0; j < N; ++j) {
		PrintToStream(fState.GetOStream(),"\n axis %lu \n", j);
		PrintToStream(fState.GetOStream(),"      x   g\n");

		//fState.GetOStream() << boost::format("\n axis %lu \n") % j ;
		//fState.GetOStream() << boost::format("      x   g\n");

		for (i = 0; i < fState.GetNBins(); i++) {
			PrintToStream(fState.GetOStream(),"weight [%11.2e , %11.2e] = ", GetCoordinate(i, j) , GetCoordinate(i + 1, j));
			PrintToStream(fState.GetOStream()," %11.2e\n", GetDistributionValue(i, j));

			//fState.GetOStream() << boost::format("weight [%11.2e , %11.2e] = ")
				//				% GetCoordinate(i, j) % GetCoordinate(i + 1, j);
			//fState.GetOStream() << boost::format(" %11.2e\n") % GetDistributionValue(i, j);
		}
		fState.GetOStream() << std::endl ;
	}
	fState.GetOStream() << std::endl ;

}

template< size_t N, hydra::detail::Backend  BACKEND , typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::PrintGrid()   {

/*
	if (!fState.GetVerbose())
		return;*/

	for (size_t j = 0; j < N; ++j) {
		PrintToStream(fState.GetOStream(),"\n axis %lu \n", j);

	//fState.GetOStream() << boost::format("\n axis %lu \n") % j ;
	fState.GetOStream() << "      x   \n";
	for (size_t i = 0; i <= fState.GetNBins(); i++) {
		PrintToStream(fState.GetOStream(), "%11.2e", GetCoordinate(i, j));
		//fState.GetOStream() << boost::format("%11.2e") % GetCoordinate(i, j);
		if (i % 5 == 4) fState.GetOStream() << std::endl;
	}
	fState.GetOStream() << std::endl;
	}
	fState.GetOStream() << std::endl;


}

template< size_t N, hydra::detail::Backend  BACKEND , typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::InitGrid() {

	GReal_t vol = 1.0;

	fState.SetNBins(1);

	for (size_t j = 0; j < N; j++) {
		GReal_t dx = fState.GetXUp()[j] - fState.GetXLow()[j];
		fState.SetDeltaX(j, dx);
		vol *= dx;

		SetCoordinate(0, j, 0.0);
		SetCoordinate(1, j, 1.0);
	}

	fState.SetVolume(vol);
	fState.SendGridToBackend();
}

template< size_t N, hydra::detail::Backend  BACKEND, typename GRND >
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::ResetGridValues() {

	for (size_t i = 0; i < fState.GetNBins(); i++) {
		for (size_t j = 0; j < N; j++) {
			SetDistributionValue(i, j, 0.0);
		}
	}
}

template< size_t N, hydra::detail::Backend  BACKEND , typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::InitBoxCoordinates() {

	//for (size_t i = 0; i < N; i++)
		//fState.SetBox(i, 0);
}


template< size_t N, hydra::detail::Backend  BACKEND, typename GRND >
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::ResizeGrid(const GInt_t bins) {



		/* weight is ratio of bin sizes */

		GReal_t pts_per_bin = (GReal_t) fState.GetNBins() / (GReal_t) bins;

		for (size_t j = 0; j < N; j++) {
			GReal_t xold;
			GReal_t xnew = 0;
			GReal_t dw = 0;
			GInt_t i = 1;

			for ( size_t k = 1; k <=  fState.GetNBins(); k++) {
				dw += 1.0;
				xold = xnew;
				xnew = GetCoordinate(k, j);

				for (; dw > pts_per_bin; i++) {
					dw -= pts_per_bin;
					SetNewCoordinate(i, xnew - (xnew - xold) * dw);
				}
			}

			for ( int  k = 1; k < bins; k++) {
				SetCoordinate(k, j, GetNewCoordinate(k));
			}

			SetCoordinate( bins, j, 1);
		}

		fState.SetNBins(bins);

}

template< size_t N , hydra::detail::Backend  BACKEND, typename GRND>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::RefineGrid() {


		for (size_t j = 0; j < N; j++) {

			GReal_t grid_tot_j, tot_weight;
			//GReal_t *weight = fState.GetWeight();

			GReal_t oldg = GetDistributionValue(0, j);
			GReal_t newg = GetDistributionValue(1, j);

			SetDistributionValue(0, j, (oldg + newg) / 2);
			grid_tot_j =  GetDistributionValue(0, j);


			for (size_t i = 1; i < fState.GetNBins() - 1; i++) {

				GReal_t rc = oldg + newg;

				oldg = newg;

				newg = GetDistributionValue(i + 1, j);

				SetDistributionValue( i, j, (rc + newg) / 3);

				grid_tot_j += GetDistributionValue(i, j);
			}

			SetDistributionValue( fState.GetNBins() - 1, j, (newg + oldg) / 2);

			grid_tot_j += GetDistributionValue(fState.GetNBins() - 1, j);

			tot_weight = 0;

			for (size_t i = 0; i < fState.GetNBins(); i++) {
				fState.SetWeight(i,0);

				if (GetDistributionValue( i, j)> 0)
				{
					oldg = grid_tot_j / GetDistributionValue( i, j);
					/* damped change */
					fState.SetWeight(i,pow (((oldg - 1) / oldg / log (oldg)), fState.GetAlpha() ));
				}

				tot_weight += fState.GetWeight()[i];

	#ifdef DEBUG
				PrintToStream(fState.GetOStream(),"weight[%d] = %g\n",i ,fState.GetWeight()[i]) ;
				//fState.GetOStream() << ("weight[%d] = %g\n") % i % fState.GetWeight()[i] ;
	#endif
			}

			{
				GReal_t pts_per_bin = tot_weight / fState.GetNBins();

				GReal_t xold;
				GReal_t xnew = 0;
				GReal_t dw = 0;
				size_t i = 1;

				for (size_t k = 0; k < fState.GetNBins(); k++) {
					dw += fState.GetWeight()[k];
					xold = xnew;
					xnew = GetCoordinate( k + 1, j);

					for (; dw > pts_per_bin; i++) {
						dw -= pts_per_bin;
						SetNewCoordinate( i,  xnew - (xnew - xold) * dw / fState.GetWeight()[k] );
					}
				}

				for (size_t k = 1; k < fState.GetNBins(); k++) {

					SetCoordinate(k, j, GetNewCoordinate(k));
				}

				SetCoordinate(fState.GetNBins(), j, 1.0);
			}
		}

}

template< size_t N, hydra::detail::Backend  BACKEND , typename GRND>
template<typename FUNCTOR>
void Vegas<N,hydra::detail::BackendPolicy<BACKEND>, GRND >::ProcessFuncionCalls(FUNCTOR const& fFunctor, GBool_t training, GReal_t& integral, GReal_t& tss)
{
	typedef hydra::detail::BackendPolicy<BACKEND> system_t;
	size_t ncalls = fState.GetCalls(training);
	size_t nkeys  = N*fState.GetCalls(training);

	// create iterators
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> first(0);
	HYDRA_EXTERNAL_NS::thrust::counting_iterator<size_t> last = first + ncalls;


	fState.CopyStateToDevice();


	detail::ResultVegas init = detail::ResultVegas();
	detail::ResultVegas result = HYDRA_EXTERNAL_NS::thrust::transform_reduce(system_t(), first, last,
			detail::ProcessCallsVegas<FUNCTOR,N,system_t ,rvector_iterator,
			uvector_iterator , GRND>(ncalls, fState, fGlobalBinInput.begin(),fFValInput.begin(), fFunctor)
	, init,	detail::ProcessBoxesVegas());


	/*
	for(size_t i=0; i< HYDRA_EXTERNAL_NS::thrust::distance( fGlobalBinInput.begin(),fGlobalBinInput.end() ); i++)
			std::cout <<"<< " << i << " " << fGlobalBinInput[i] << " " << fFValInput[i] << std::endl;
	 */

	HYDRA_EXTERNAL_NS::thrust::sort_by_key(system_t(),fGlobalBinInput.begin(),fGlobalBinInput.end(), fFValInput.begin());
	/*
	for(size_t i=0; i< HYDRA_EXTERNAL_NS::thrust::distance( fGlobalBin.begin(),fGlobalBin.end() ); i++)
		std::cout <<"<< " << i << " " << fGlobalBin[i] << " " << fFVal[i] << std::endl;
	 */

	auto end_iterators = HYDRA_EXTERNAL_NS::thrust::reduce_by_key(system_t(),fGlobalBinInput.begin(),fGlobalBinInput.end(),fFValInput.begin(),
			fGlobalBinOutput.begin(),  fFValOutput.begin());



	HYDRA_EXTERNAL_NS::thrust::copy( fFValOutput.begin(), end_iterators.second, fState.GetDistribution().begin());
	integral=result.fMean*result.fN  ;
	tss=sqrt( result.fM2 );


}


}

//#endif /* VEGAS_INL_ */
