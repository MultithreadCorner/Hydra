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
 * VegasState.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */


#ifndef VEGASSTATE_INL_
#define VEGASSTATE_INL_

namespace hydra {

template<size_t N , hydra::detail::Backend BACKEND>
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::VegasState(std::array<GReal_t,N> const& xlower,
		std::array<GReal_t,N> const& xupper) :
		fTrainingIterations(1),
		fTrainedGridFrozen(0),
		fNDimensions(N),
		fNBinsMax(BINS_MAX),
		fNBins(BINS_MAX),
		fNBoxes(0),
		fVolume(0),
		fAlpha(1.5),
		fMode(MODE_IMPORTANCE),
		fVerbose(-1),
		fIterations(5),
		fStage(0),
		fJacobian(0),
		fWeightedIntSum(0),
		fSumOfWeights(0),
		fChiSum(0),
		fChiSquare(0),
		fResult(0),
		fSigma(10),
		fItStart(0),
		fItNum(0),
		fSamples(0),
		fCallsPerBox(0),
		fCalls(5000),
		fTrainingCalls(5000),
		fMaxError(0.5e-3),
		fUseRelativeError(kTrue),
		fOStream(std::cout),
		//-------
		fBackendXLow(N),
		fBackendDeltaX(N),
		//fBackendDistribution(N * BINS_MAX),
		fBackendXi((BINS_MAX + 1) * N),
		//-------
		fDistribution(N * BINS_MAX),
		fDeltaX(N),
		fXi((BINS_MAX + 1) * N),
		fXin(BINS_MAX + 1),
		fWeight(BINS_MAX),
		fXUp(N),
		fXLow(N)
{

	for(size_t i=0; i<N; i++)
	{
		fXUp[i]=xupper[i];
		fXLow[i]=xlower[i];
		fDeltaX[i] = xupper[i]-xlower[i];
		fBackendDeltaX[i]= xupper[i]-xlower[i];
	}


}


template<size_t N , hydra::detail::Backend BACKEND>
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::VegasState(const GReal_t xlower[N], const GReal_t xupper[N]) :
		fTrainingIterations(1),
		fTrainedGridFrozen(0),
		fNDimensions(N),
		fNBinsMax(BINS_MAX),
		fNBins(BINS_MAX),
		fNBoxes(0),
		fVolume(0),
		fAlpha(1.5),
		fMode(MODE_IMPORTANCE),
		fVerbose(-1),
		fIterations(5),
		fStage(0),
		fJacobian(0),
		fWeightedIntSum(0),
		fSumOfWeights(0),
		fChiSum(0),
		fChiSquare(0),
		fResult(0),
		fSigma(10),
		fItStart(0),
		fItNum(0),
		fSamples(0),
		fCallsPerBox(0),
		fCalls(5000),
		fTrainingCalls(5000),
		fMaxError(0.5e-3),
		fUseRelativeError(kTrue),
		fOStream(std::cout),
		//-------
		fBackendXLow(N),
		fBackendDeltaX(N),
		//fBackendDistribution(N * BINS_MAX),
		fBackendXi((BINS_MAX + 1) * N),
		//-------
		fDistribution(N * BINS_MAX),
		fDeltaX(N),
		fXi((BINS_MAX + 1) * N),
		fXin(BINS_MAX + 1),
		fWeight(BINS_MAX),
		fXUp(N),
		fXLow(N)
{

	for(size_t i=0; i<N; i++)
	{
		fXUp[i]=xupper[i];
		fXLow[i]=xlower[i];
		fDeltaX[i] = xupper[i]-xlower[i];
		fBackendDeltaX[i]= xupper[i]-xlower[i];
	}


}


template<size_t N , hydra::detail::Backend BACKEND>
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::VegasState(VegasState<N,hydra::detail::BackendPolicy<BACKEND>> const& other) :
        fTrainingIterations(other.GetTrainingIterations()),
        fTrainedGridFrozen(other.IsTrainedGridFrozen()),
		fAlpha(other.GetAlpha()),
		fNDimensions(other.GetNDimensions()),
		fNBinsMax(other.GetNBinsMax()),
		fNBins(other.GetNBins()),
		fNBoxes(other.GetNBoxes()),
		fVolume(other.GetVolume()),
		fMode(other.GetMode()),
		fVerbose(other.GetVerbose()),
		fIterations(other.GetIterations()),
		fStage(other.GetStage()),
		fJacobian(other.GetJacobian()),
		fWeightedIntSum(other.GetWeightedIntSum()),
		fSumOfWeights(other.GetSumOfWeights()),
		fChiSum(other.GetChiSum()),
		fChiSquare(other.GetChiSquare()),
		fResult(other.GetResult()),
		fSigma(other.GetSigma()),
		fItStart(other.GetItStart()),
		fItNum(other.GetItNum()),
		fSamples(other.GetSamples()),
		fMaxError(other.GetMaxError()),
		fUseRelativeError(other.IsUseRelativeError()),
		fCallsPerBox(other.GetCallsPerBox()),
		fCalls(other.GetCalls()),
		fTrainingCalls(other.GetTrainingCalls()),
		fDeltaX(other.GetDeltaX()),
		fDistribution(other.GetDistribution()),
		fXi(other.GetXi()),
		fXin(other.GetXin()),
		fWeight(other.GetWeight()),
		fXLow(other.GetXLow()),
		fXUp(other.GetXUp()),
		fIterationResult(other.GetIterationResult()),
		fIterationSigma(other.GetIterationSigma()),
		fCumulatedResult(other.GetCumulatedResult()),
		fCumulatedSigma(other.GetCumulatedSigma()),
		fIterationDuration(other.GetIterationDuration()),
		fBackendDeltaX(other.GetBackendDeltaX()),
		fBackendXi(other.GetBackendXi()),
		fBackendXLow(other.GetBackendXLow()),
		//fBackendDistribution(other.GetBackendDistribution()),
		fOStream(std::cout) {}

template<size_t N , hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2>
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::
VegasState( VegasState<N, hydra::detail::BackendPolicy <BACKEND2>> const& other) :
        fTrainingIterations(other.GetTrainingIterations()),
        fTrainedGridFrozen(other.IsTrainedGridFrozen()),
		fAlpha(other.GetAlpha()),
		fNDimensions(other.GetNDimensions()),
		fNBinsMax(other.GetNBinsMax()),
		fNBins(other.GetNBins()),
		fNBoxes(other.GetNBoxes()),
		fVolume(other.GetVolume()),
		fMode(other.GetMode()),
		fVerbose(other.GetVerbose()),
		fIterations(other.GetIterations()),
		fStage(other.GetStage()),
		fJacobian(other.GetJacobian()),
		fWeightedIntSum(other.GetWeightedIntSum()),
		fSumOfWeights(other.GetSumOfWeights()),
		fChiSum(other.GetChiSum()),
		fChiSquare(other.GetChiSquare()),
		fResult(other.GetResult()),
		fSigma(other.GetSigma()),
		fItStart(other.GetItStart()),
		fItNum(other.GetItNum()),
		fSamples(other.GetSamples()),
		fMaxError(other.GetMaxError()),
		fUseRelativeError(other.IsUseRelativeError()),
		fCallsPerBox(other.GetCallsPerBox()),
		fCalls(other.GetCalls()),
		fTrainingCalls(other.GetTrainingCalls()),
		fDeltaX(other.GetDeltaX()),
		fDistribution(other.GetDistribution()),
		fXi(other.GetXi()),
		fXin(other.GetXin()),
		fWeight(other.GetWeight()),
		fXLow(other.GetXLow()),
		fXUp(other.GetXUp()),
		fIterationResult(other.GetIterationResult()),
		fIterationSigma(other.GetIterationSigma()),
		fCumulatedResult(other.GetCumulatedResult()),
		fCumulatedSigma(other.GetCumulatedSigma()),
		fIterationDuration(other.GetIterationDuration()),
		fBackendDeltaX(other.GetBackendDeltaX()),
		fBackendXi(other.GetBackendXi()),
		fBackendXLow(other.GetBackendXLow()),
		//fBackendDistribution(other.GetBackendDistribution()),
		fOStream(std::cout) {}



template<size_t N , hydra::detail::Backend BACKEND>
VegasState<N,hydra::detail::BackendPolicy<BACKEND>>&
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::operator=(VegasState<N,hydra::detail::BackendPolicy<BACKEND>> const& other)
{
	if(this==&other)return *this;
        fTrainingIterations=other.GetTrainingIterations();
        fTrainedGridFrozen=other.IsTrainedGridFrozen();
		fAlpha=other.GetAlpha();
		fNDimensions=other.GetNDimensions();
		fNBinsMax=other.GetNBinsMax();
		fNBins=other.GetNBins();
		fNBoxes=other.GetNBoxes();
		fVolume=other.GetVolume();
		fMode=other.GetMode();
		fVerbose=other.GetVerbose();
		fIterations=other.GetIterations();
		fStage=other.GetStage();
		fJacobian=other.GetJacobian();
		fWeightedIntSum=other.GetWeightedIntSum();
		fSumOfWeights=other.GetSumOfWeights();
		fChiSum=other.GetChiSum();
		fChiSquare=other.GetChiSquare();
		fResult=other.GetResult();
		fSigma=other.GetSigma();
		fItStart=other.GetItStart();
		fItNum=other.GetItNum();
		fSamples=other.GetSamples();
		fMaxError=other.GetMaxError();
		fUseRelativeError=other.IsUseRelativeError();
		fCallsPerBox=other.GetCallsPerBox();
		fCalls=other.GetCalls();
		fTrainingCalls=other.GetTrainingCalls();
		fDeltaX=other.GetDeltaX();
		fDistribution=other.GetDistribution();
		fXi=other.GetXi();
		fXin=other.GetXin();
		fWeight=other.GetWeight();
		fXLow=other.GetXLow();
		fXUp=other.GetXUp();
		fIterationResult=other.GetIterationResult();
		fIterationSigma=other.GetIterationSigma();
		fCumulatedResult=other.GetCumulatedResult();
		fCumulatedSigma=other.GetCumulatedSigma();
		fIterationDuration=other.GetIterationDuration();
		fBackendDeltaX=other.GetBackendDeltaX();
		fBackendXi=other.GetBackendXi();
		fBackendXLow=other.GetBackendXLow();
		//fBackendDistribution=other.GetBackendDistribution();

		fOStream =other.GetOStream();

		return *this;

}

template<size_t N , hydra::detail::Backend BACKEND>
template<hydra::detail::Backend BACKEND2>
VegasState<N,hydra::detail::BackendPolicy<BACKEND>>&
VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::operator=( VegasState<N, hydra::detail::BackendPolicy <BACKEND2>> const& other)
{
	if(this==&other)return *this;

        fTrainingIterations=other.GetTrainingIterations();
        fTrainedGridFrozen=other.IsTrainedGridFrozen();
		fAlpha=other.GetAlpha();
		fNDimensions=other.GetNDimensions();
		fNBinsMax=other.GetNBinsMax();
		fNBins=other.GetNBins();
		fNBoxes=other.GetNBoxes();
		fVolume=other.GetVolume();
		fMode=other.GetMode();
		fVerbose=other.GetVerbose();
		fIterations=other.GetIterations();
		fStage=other.GetStage();
		fJacobian=other.GetJacobian();
		fWeightedIntSum=other.GetWeightedIntSum();
		fSumOfWeights=other.GetSumOfWeights();
		fChiSum=other.GetChiSum();
		fChiSquare=other.GetChiSquare();
		fResult=other.GetResult();
		fSigma=other.GetSigma();
		fItStart=other.GetItStart();
		fItNum=other.GetItNum();
		fSamples=other.GetSamples();
		fMaxError=other.GetMaxError();
		fUseRelativeError=other.IsUseRelativeError();
		fCallsPerBox=other.GetCallsPerBox();
		fCalls=other.GetCalls();
		fTrainingCalls=other.GetTrainingCalls();
		fDeltaX=other.GetDeltaX();
		fDistribution=other.GetDistribution();
		fXi=other.GetXi();
		fXin=other.GetXin();
		fWeight=other.GetWeight();
		fXLow=other.GetXLow();
		fXUp=other.GetXUp();
		fIterationResult=other.GetIterationResult();
		fIterationSigma=other.GetIterationSigma();
		fCumulatedResult=other.GetCumulatedResult();
		fCumulatedSigma=other.GetCumulatedSigma();
		fIterationDuration=other.GetIterationDuration();
		fBackendDeltaX=other.GetBackendDeltaX();
		fBackendXi=other.GetBackendXi();
		fBackendXLow=other.GetBackendXLow();
		//fBackendDistribution=other.GetBackendDistribution();
		fOStream =other.GetOStream();

		return *this;
}



template<size_t N , hydra::detail::Backend BACKEND>
void VegasState<N, hydra::detail::BackendPolicy<BACKEND>>::ClearStoredIterations()
{
	fIterationResult.clear();
	fIterationSigma.clear();
	fCumulatedResult.clear();
	fCumulatedSigma.clear();
	fIterationDuration.clear();
	fFunctionCallsDuration.clear();

}
}

#endif /* VEGASSTATE_INL_ */
