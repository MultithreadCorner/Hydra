/*
 * TestHydra.cu
 *
 *  Created on: Jun 21, 2016
 *      Author: augalves
 */


#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include <time.h>
#include <string>
#include <map>
#include <vector>
#include <array>
#include <tuple>
#include <chrono>
#include <type_traits>
#include <typeinfo>
//command line
#include <tclap/CmdLine.h>
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

//this lib
#include <hydra/Types.h>
#include <hydra/Vector4R.h>
#include <hydra/PhaseSpace.h>
#include <hydra/Containers.h>
#include <hydra/Evaluate.h>
#include <hydra/Function.h>
#include <hydra/FunctorArithmetic.h>
#include <hydra/FunctionWrapper.h>
#include <hydra/Random.h>
#include <hydra/VegasState.h>
#include <hydra/Vegas.h>
#include <hydra/Plain.h>
#include <hydra/LogLikelihoodFCN.h>
#include <hydra/PointVector.h>
#include <hydra/Parameter.h>
#include <hydra/UserParameters.h>
#include <hydra/Pdf.h>
#include <hydra/AddPdf.h>

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/MnPlot.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include "Minuit2/VariableMetricMinimizer.h"
//root
#include <TROOT.h>
#include <TH1D.h>
#include <TF1.h>
#include <TH2D.h>
#include <TH3D.h>
#include <TApplication.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TString.h>
#include <TStyle.h>
#include "RooGlobalFunc.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooCFunction1Binding.h"
#include "RooTFnBinding.h"
#include "RooPlot.h"

#include <src/Gauss.h>

#include <src/Gauss.h>
#include <src/Exp.h>

using namespace std;
using namespace ROOT::Minuit2;
using namespace hydra;


GInt_t main(int argv, char** argc)
{


	size_t nentries = 0;

	try {

		TCLAP::CmdLine cmd("Command line arguments for HydraRandomExample", '=');

		TCLAP::ValueArg<GULong_t> EArg("n", "number-of-events",
				"Number of events",
				true, 5e6, "long");
		cmd.add(EArg);

		// Parse the argv array.
		cmd.parse(argv, argc);

		// Get the value parsed by each arg.
		nentries = EArg.getValue();

	}
	catch (TCLAP::ArgException &e)  {
		std::cerr << "error: " << e.error() << " for arg " << e.argId()
														<< std::endl;
	}
	//Print::SetLevel(0);
	//ROOT::Minuit2::MnPrint::SetLevel(2);
	//----------------------------------------------

	//Generator with current time count as seed.
	size_t seed = std::chrono::system_clock::now().time_since_epoch().count();
	Random<thrust::random::default_random_engine> Generator( seed  );


	//-------------------------------------------
	//range of the analysis
	std::array<GReal_t, 1>  min   ={ 0.0};
	std::array<GReal_t, 1>  max   ={ 15.0};

	//------------------------------------
	string Mean1("Mean1");
	string Sigma1("Sigma1");
	string Mean2("Mean2");
	string Sigma2("Sigma2");
	string Tau("Tau");

	//fit paremeters
/*
	Variable  mean1( Mean1 , -2.0, 0.001, -2.5, -1.5); // mean value 0, index 0
	Variable  sigma1(Sigma1 , 1.0, 0.001,  0.5, 1.5);

	Variable  mean2( Mean2 ,  2.0, 0.001,  1.5, 2.5); // mean value 0, index 0
	Variable  sigma2(Sigma2 , 1.0, 0.001,  0.5, 1.5);
*/

	MnUserParameters upp;
	upp.Add("mean", 1, 0.1);

	Parameter  mean1  = Parameter::Create().Name(Mean1).Value(3.0) .Error(0.001).Limits( 1, 4);
	Parameter  sigma1 = Parameter::Create().Name(Sigma1).Value(0.5).Error(0.001).Limits(0.1, 1.5);
	Parameter  mean2  = Parameter::Create().Name(Mean2).Value(5.0).Error(0.001).Limits(4, 6);
	Parameter  sigma2 = Parameter::Create().Name(Sigma2).Value(1.0).Error(0.001).Limits(0.5, 1.5);
    Parameter  tau    = Parameter::Create().Name(Tau).Value(1).Error(0.001).Limits( -2, 2);

	UserParameters upar;
	upar.AddParameter(&mean1);
	upar.AddParameter(&sigma1);
	upar.AddParameter(&mean2);
	upar.AddParameter(&sigma2);
	upar.AddParameter(&tau);

	/*
	upar.Fix("Mean1" );
	upar.Fix("Sigma1" );

	upar.Fix("Mean2");
	upar.Fix("Sigma2" );*/


	Gauss GaussianA(mean1, sigma1,0);
	Gauss GaussianB(mean2, sigma2,0);
	Exp   Exponential(tau);

	//integration
    //Vegas state hold the resources for performing the integration
    VegasState<1> *state = new VegasState<1>( min, max); // nota bene: the same range of the analisys
	state->SetVerbose(-1);
	state->SetAlpha(1.75);
	state->SetIterations(5);
	state->SetUseRelativeError(1);
	state->SetMaxError(1e-3);
    //10,000 call (fast convergence and very precice)
	Vegas<1> vegas( state,10000);

	auto GaussianA_PDF   = make_pdf(GaussianA, &vegas);
	auto GaussianB_PDF   = make_pdf(GaussianB, &vegas);
	auto Exponential_PDF = make_pdf(Exponential, &vegas);

	GaussianA_PDF.PrintRegisteredParameters();

	vegas.Integrate(GaussianA_PDF);
	cout << ">>> GaussianA intetgral prior fit "<< endl;
	cout << "Result: " << vegas.GetResult() << " +/- "
		 << vegas.GetAbsError() << " Chi2: "<< state->GetChiSquare() << endl;


	GaussianB_PDF.PrintRegisteredParameters();

	vegas.Integrate(GaussianB_PDF);
	cout << ">>> GaussianB intetgral prior fit "<< endl;
	cout << "Result: " << vegas.GetResult() << " +/- "
			<< vegas.GetAbsError() << " Chi2: "<< state->GetChiSquare() << endl;

	Exponential_PDF.PrintRegisteredParameters();

		vegas.Integrate(Exponential_PDF);
		cout << ">>> Exponential intetgral prior fit "<< endl;
		cout << "Result: " << vegas.GetResult() << " +/- "
				<< vegas.GetAbsError() << " Chi2: "<< state->GetChiSquare() << endl;



	std::string na("NA");
	std::string nb("NB");
	std::string nc("NC");

	Parameter NA(na ,nentries+10, nentries/1000, (nentries)/10, 2*nentries) ;
	Parameter NB(nb ,nentries+10, nentries/1000, (nentries)/10, 2*nentries) ;
	Parameter NC(nc ,nentries+10, nentries/1000, (nentries)/10, 2*nentries) ;


	upar.AddParameter(&NA);
	upar.AddParameter(&NB);
	upar.AddParameter(&NC);



	std::array<Parameter*, 3>  yields  ={&NA, &NB, &NC};

	auto model = add_pdfs(yields, GaussianA_PDF, GaussianB_PDF, Exponential_PDF );


	//return 0;
	//-----------------------------------
	//Generate data
	PointVector<device, GReal_t, 1> data_d(3*nentries);

	Generator.Gauss(mean1 , sigma1, data_d.begin(), data_d.begin() + nentries );
	Generator.Gauss(mean2 , sigma2, data_d.begin()+ nentries, data_d.begin() + 2*nentries );
	Generator.Uniform(min[0], max[0], data_d.begin()+ 2*nentries, data_d.end() );


	//---------------------------
	//get data from device and fill histogram
	PointVector<host> data_h(data_d);

	TH1D hist_gaussian("gaussian", "", 100, min[0], max[0]);

	for(auto point: data_h )
		hist_gaussian.Fill(point.GetCoordinate(0));


	//-------------------------------------------------
	//get the FCN
	auto modelFCN = make_loglikehood_fcn(model, data_d.begin(), data_d.end() );


	cout << upar << endl;


	MnStrategy strategy(2);


	// create Migrad minimizer
	MnMigrad migrad(modelFCN,upar.GetState() ,  strategy);


	// Minimize

	// ... and Minimize
	FunctionMinimum minimum = migrad(500000*6, 10);
	//return 0
	// output
	std::cout<<"minimum: "<<minimum<<std::endl;


/*
	MnMinos Minos(modelFCN, minimum, strategy);

	// 1-sigma MINOS errors (minimal interface)
	modelFCN.SetErrorDef(0.5);

	std::vector<MinosError> errors;

	for(size_t i=0; i<upar.Params().size(); i++ )
		errors.push_back(Minos.Minos(i, 5000*6, 100));

	std::cout<<  "================================================================================"<< endl;
	// output
	for(size_t i=0; i<upar.Params().size(); i++ )
	{
		std::cout<<  errors[i] << endl;
	}
	std::cout<<  "================================================================================"<< endl;

*/

    //Set the function with the fitted parameters
	model.SetParameters(minimum.UserParameters().Params());
	model.PrintRegisteredParameters();

	//sample fit function on the host this time
	PointVector<host, GReal_t, 1> data2_h(0);
	Generator.SetSeed(std::chrono::system_clock::now().time_since_epoch().count()+1);
	Generator.Sample<host>(model, min, max, data2_h, nentries/2 );

	TH1D hist_gaussian_fit("gaussian_fit", "", 100, min[0], max[0]);

	// histogram it for graphics representation
	for(auto point: data2_h )
			hist_gaussian_fit.Fill(point.GetCoordinate(0));

	TH1D hist_gaussian_plot("gaussian_plot", "", 100,  min[0], max[0]);
	for (size_t i=1 ; i<=200 ; i++) {
		GReal_t x = hist_gaussian_plot.GetBinCenter(i);
		hist_gaussian_plot.SetBinContent(i, model(&x) );
	}

	//scale
	hist_gaussian_fit.Scale(hist_gaussian.Integral()/hist_gaussian_fit.Integral() );
	hist_gaussian_plot.Scale(hist_gaussian.Integral()/hist_gaussian_plot.Integral() );



	/***********************************
	 * RootApp, Drawing...
	 ***********************************/

	TApplication *myapp=new TApplication("myapp",0,0);


	TCanvas canvas_gauss("canvas_gauss", "Gaussian distribution", 500, 500);
	hist_gaussian.Draw("e0");
	hist_gaussian.SetMarkerSize(1);
	hist_gaussian.SetMarkerStyle(20);

	//sampled data after fit
	hist_gaussian_fit.Draw("barSAME");
	hist_gaussian_fit.SetLineColor(4);
	hist_gaussian_fit.SetFillColor(4);
	hist_gaussian_fit.SetFillStyle(3001);

	//original data
	hist_gaussian.Draw("e0SAME");

	//plot
	hist_gaussian_plot.Draw("lfsame");
	hist_gaussian_plot.SetLineColor(2);
	hist_gaussian_plot.SetLineWidth(2);

	canvas_gauss.Print("Fit_gaussian.pdf");

	myapp->Run();

	return 0;

}

