Phase-space Monte Carlo
=======================

Phase-Space Monte Carlo simulates the kinematics of a particle with a given four-momentum
decaying to a n-particle final state, without intermediate resonances. Samples 
of phase-space Monte Carlo events are widely used in HEP studies where 
the calculation of phase-space volume is required as well as a starting point to implement and describe the properties of models with one or more resonances or even to simulate the response of the detector to decay's products [James]_. 

Hydra provides an implementation of the Raubold-Lynch method [James]_
and can generate the full kinematics of decays with any number of particles in the final state.
Sequential decays, evaluation of models, production of weighted and unweighted samples and many other features are also supported.


Decays and decay chains
-----------------------

The four-vector of the generated final-state particles are stored in the dedicated vector-like container ``hydra::Decays<N,BACKEND>`` 
where ``N`` is the number of particles in the final state and ``BACKEND``is the memory space where allocate the storage. ``hydra::Decays<N,BACKEND>`` can be aggregated to describe sequential decays using ``hydra::Chains<Decays...>`` objects.

Both classes are iterable, but the ``hydra::Chains<Decays...>`` container does not implement a full vector-like interface. Pre-allocated ``hydra::Decays<N,BACKEND>`` can not be added to a 
``hydra::Chains<Decays...>``.

Phase-space Monte Carlo generator.
----------------------------------

The phase-space Monte Carlo generator is represented by the class ``hydra::PhaseSpace<N,RNG>``, where is the number of particles and ``RGN`` is underlying random number generator to be used. 
The constructor of the ``hydra::PhaseSpace`` takes as parameter an array with the masses of the final state particles.  The decays are generated invoking the overloaded 
``hydra::PhaseSpace::Generate(...)`` method. This method can take a ``hydra::Vector4R``, describing momentum of a only mother particle or iterators pointing for a container storing a list of mother particles and the iterators pointing to the ``hydra::Decays<N,BACKEND>`` container that will hold the generated final states. If an explicit policy policy is passed, the generation is parallelized in the corresponding back-end, otherwise the class will process the random number generation in the back-end where the containers are allocated.

Generating one-level decays
...........................

The code below shows how to generate a sample of 20 million :math:`B^0 \to J/\psi K^+ \pi^-` decays and fill a Dalitz's plot, i.e. a histogram  :math:`M^2( J/\psi \pi^-) vs M^2(K^+ \pi^-)`:

.. code-block:: cpp

	#include <hydra/Types.h>
	#include <hydra/Vector4R.h>
	#include <hydra/PhaseSpace.h>
	#include <hydra/device/System.h>
	#include <hydra/Decays.h>

	...

	size_t  nentries  = 20e6;       // number of events to generate
	double B0_mass    = 5.27955;    // B0 mass
	double Jpsi_mass  = 3.0969;     // J/psi mass
	double K_mass     = 0.493677;   // K+ mass
	double pi_mass    = 0.13957061; // pi mass

	// mother particle
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// decays container
	hydra::Decays<3, hydra::device::sys_t > Events(nentries);

	hydra::PhaseSpace<3> phsp{Jpsi_mass, K_mass, pi_mass};

	// generate the final state particles
	phsp.Generate(B0, Events.begin(), Events.end());

	// functor to calculate Dalitz variables
	auto dalitz_calculator = hydra::wrap_lambda( 
		[=] __host__ __device__ (unsigned int np, hydra::Vector* particles){

			hydra::Vector4R Jpsi = event[0];
			hydra::Vector4R K    = event[1];
			hydra::Vector4R pi   = event[2];

			double M2_Jpsi_pi = (Jpsi + pi).mass2();
			double M2_Kpi     = (K + pi).mass2();

			return hydra::make_tuple( M2_Jpsi_pi, M2_Kpi);	
		}
	);

	// 
	auto particles        = Events.GetUnweightedDecays();

	// use an smart-range to calculate the Dalitz variables
	// without have to store it. ;)
	auto dalitz_variables = hydra::make_range( particles.begin(), particles.end(), dalitz_calculator);

	// get the event's weights
	auto dalitz_weights   = Events.GetWeights();

	// instantiate 2D histogram
	hydra::DenseHistogram<2, double> Hist_Dalitz({100,100}, {pow(Jpsi_mass + pi_mass,2), pow(K_mass + pi_mass,2)},
		{pow(B0_mass - K_mass,2)   , pow(B0_mass - Jpsi_mass,2)} );

	//fill the histogram
	Hist_Dalitz.Fill(dalitz_variables.begin(), dalitz_variables.end(), dalitz_weights.begin());

	...

In the previous example, the user can foward the ``Hydra::DenseHistogram`` to ROOT and draw it. 


Generating sequential decays
.............................


The code below shows how to generate a sample of 20 million decay chains :math:`B^0 \to J/\psi K^+ \pi^-` with :math:`J/\psi \to \mu^+ \mu^-`. 

The first step to process the decay chain is to the generate the decays  :math:`B^0 \to J/\psi K^+ \pi^-`, then the list of :math:`J/\psi` candidates is passed to the instance of ``hydra::PhaseSpace`` to generate the  :math:`J/\psi \to \mu^+ \mu^-` corresponding to each :math:`J/\psi` mother. 
Notice that the decay events stored in a given chain are accessed using a ``hydra::placeholder``.  

.. code-block:: cpp

	#include <hydra/Types.h>
	#include <hydra/Vector4R.h>
	#include <hydra/PhaseSpace.h>
	#include <hydra/device/System.h>
	#include <hydra/Chains.h>
	#include <hydra/Placeholders.h>
	
	...

	using namespace hydra::placeholders;
	
	...

	size_t nentries   = 20e6;         // number of events to generate
	double B0_mass    = 5.27955;      // B0 mass
	double Jpsi_mass  = 3.0969;       // J/psi mass
	double K_mass     = 0.493677;     // K+ mass
	double pi_mass    = 0.13957061;   // pi mass
	double mu_mass    = 0.1056583745 ;// mu mass


	// mother particle
	hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);

	// create PhaseSpace object for B0 -> K pi J/psi
	hydra::PhaseSpace<3> phsp_B2JpsiKpi{Jpsi_mass, K_mass, pi_mass };

	// create PhaseSpace object for J/psi -> mu+ mu-
	hydra::PhaseSpace<2> phsp_Jpsi2mumu{mu_mass , mu_mass};

	//	allocate memory to hold the final states particles
	auto Events = hydra::make_chain<3,2>(hydra::device::sys, nentries);

	//generate the final state particles for B0 -> K pi J/psi
	phsp_B2JpsiKpi.Generate(B0, Events.GetDecay(_0).begin(),
				Events.GetDecay(_0).end());

	//pass the list of J/psi to generate the final
	//state particles for J/psi -> mu+ mu-
	phsp_Jpsi2mumu.Generate(Events.GetDecay(_0).GetDaughters(0).begin(),
	         Events.GetDecay(_0).GetDaughters(0).end(),
	          Events.GetDecay(_1).begin());
 


Other features
--------------

The classes of the phase-space module provides many other functionality. 
The list below summarizes some of them:

* Calculate the mean and the variance of a functor over a phase-space without the need to generate and store events. 
* Evaluate functors and stored the result without the need to generate and store events.
* Unweight and re-weight events stored in ``hydra::decay`` objects to match .
* Access single particle's ``Vector4R`` or its components of events stored in ``hydra::decay`` objects and interact with it. 

For brevity, the user is adivesed to look the doxygen documentation and the examples to learn what is available and how to deploy it. 