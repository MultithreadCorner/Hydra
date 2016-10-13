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
 * VegasState.inl
 *
 *  Created on: 21/08/2016
 *      Author: Antonio Augusto Alves Junior
 */

/**
 * \file
 * \ingroup numerical_integration
 */


#ifndef VEGASSTATE_INL_
#define VEGASSTATE_INL_

namespace hydra {

template<size_t N>
VegasState<N>::VegasState(std::array<GReal_t,N> const& xlower,
		std::array<GReal_t,N> const& xupper) :
		fNDimensions(N),
		fNBinsMax(BINS_MAX),
		fNBins(BINS_MAX),
		fNBoxes(0),
		fVolume(0),
		fAlpha(1.7),
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
		fSigma(0),
		fItStart(0),
		fItNum(0),
		fSamples(0),
		fCallsPerBox(0),
		fMaxError(0.5e-3),
		fUseRelativeError(kTrue),
		fOStream(std::cout)
{

	for(size_t i=0; i<N; i++)
	{
		fXUp.push_back(xupper[i]);
		fXLow.push_back(xlower[i]);
		fDeltaX.push_back(xupper[i]-xlower[i]);

	}
	fDeviceXLow.resize(N);
	fDeviceDeltaX.resize(N);

	fDistribution.resize(N * BINS_MAX);
	fDeviceDistribution.resize(N * BINS_MAX);

	fXi.resize((BINS_MAX + 1) * N);
	fDeviceXi.resize((BINS_MAX + 1) * N);


	fXin.resize(BINS_MAX + 1);
	fWeight.resize(BINS_MAX);

	fIterationResult.resize(0);
	fIterationSigma.resize(0);
	fCumulatedResult.resize(0);
	fCumulatedSigma.resize(0);
	fIterationDuration.resize(0);

}

template<size_t N>
VegasState<N>::VegasState(VegasState const& other) :
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
		fOStream(std::cout) {}



template<size_t N>
void VegasState<N>::ResetState()
		{

		fNDimensions= N;
		fNBinsMax = BINS_MAX;
		fNBins=BINS_MAX;
		fNBoxes=0;
		fVolume=0;
		fAlpha=1.7;
		fMode=MODE_IMPORTANCE;
		fVerbose=-1;
		fIterations=5;
		fStage=0;
		fJacobian=0;
		fWeightedIntSum=0;
		fSumOfWeights=0;
		fChiSum=0;
		fChiSquare=0;
		fResult=0;
		fSigma=0;
		fItStart=0;
		fItNum=0;
		fSamples=0;
		fCallsPerBox=0;

		thrust::fill(fDeviceXLow.begin(), fDeviceXLow.end(),  0.0);
		thrust::fill( fDeviceDeltaX.begin(), fDeviceDeltaX.end(),  0.0);
		thrust::fill( fDistribution.begin(), fDistribution.end(),  0.0);
		thrust::fill( fDeviceDistribution.begin(), fDeviceDistribution.end(),  0.0);
		thrust::fill( fXi.begin(), fXi.end(),  0.0);
		thrust::fill( fDeviceXi.begin(), fDeviceXi.end(),  0.0);
		thrust::fill( fXin.begin(), fXin.end(),  0.0);
		thrust::fill( fWeight.begin(), fWeight.end(),  0.0);
		thrust::fill( fIterationResult.begin(), fIterationResult.end(),  0.0);
		thrust::fill( fIterationSigma.begin(), fIterationSigma.end(),  0.0);
		thrust::fill( fCumulatedResult.begin(), fCumulatedResult.end(),  0.0);
		thrust::fill( fCumulatedSigma.begin(), fCumulatedSigma.end(),  0.0);
		thrust::fill( fIterationDuration.begin(), fIterationDuration.end(),  0.0);


		}
}

#endif /* VEGASSTATE_INL_ */
