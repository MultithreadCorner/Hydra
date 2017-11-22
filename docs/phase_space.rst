Phase-space Monte Carlo
=======================

Phase-Space Monte Carlo simulates the kinematics of a particle with a given four-momentum
decaying to a n-particle final state, without intermediate resonances. Samples 
of phase-space Monte Carlo events are widely used in HEP studies where 
the calculation of phase-space volume is required as well as a starting point to implement and to describe the properties of models with one or more resonances or to simulate the response of the detector to decay's products [James]_. 

Hydra provides an implementation of the Raubold-Lynch method [James]_
and can generate the full kinematics of decays with any number of particles in the final state.
Sequential decays, evaluation of model, production of weighted and unweighted samples and many other features are also supported.


Decays and decay chains
-----------------------

Decays are stored in the dedicated vector-like container ``hydra::Decays<N,BACKEND>`` 
where ``N`` is the number of particles in the final state and ``BACKEND``is the memory space where to allocate the storage. ``hydra::Decays<N,BACKEND>`` can be aggregated to describe sequential decays using ``hydra::Chains<Decays...>``.

Both classes are iterable, but the ``hydra::Chains<Decays...>`` container does not implement a full vector-like interface yet. Preallocated  ``hydra::Decays<N,BACKEND>`` can not be added to a ``hydra::Chains<Decays...>``.

Phase-space Monte Carlo generator.
----------------------------------

The phase-space Monte Carlo generator is represented by the class ``hydra::PhaseSpace<N,RNG>``, where is the number of particles and ``RGN`` is underlying random number generator to be used. 
The constructor of the ``hydra::PhaseSpace`` takes as parameter an array with the masses of the final state particles.  The decays are generated invoking the overloaded 
``hydra::PhaseSpace::Generate(...)`` method. This method can take a ``hydra::Vector4R``, describing momentum of the mother particle, or iterators pointing for a container storing a list of mother particles; and the iterators pointing to the ``hydra::Decays<N,BACKEND>`` container that will hold the generated final states. If an explicit policy policy is passed, the generation is parallelized in the corresponding back-end, otherwise the class will process the random number generation in the back-end the containers is allocated.  

The code below shows how to generate a sample of 20 million :math:`B^0 \to J/\psi K^+ \pi^-` decays and fill a Dalitz's plot, i.e. a histogram  :math:`{M^2( J/\psi \pi^-), M^2(K^+ \pi^-)}`:

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

hydra::Vector4R B0(B0_mass, 0.0, 0.0, 0.0);
double masses[3]{Jpsi_mass, K_mass, pi_mass };

//decays container
hydra::Decays<3, hydra::device::sys_t > Events(nentries);

//generate the final state particles
phsp.Generate(B0, Events.begin(), Events.end());

//functor for Dalitz variables
auto dalitz = hydra::wrap_lambda( 
	[=] __host__ __device__ ( hydra::tuple<double, hydra::Vector, hydra::Vector, hydra::Vector>& event ){

		double weight        = hydra::get<0>(event);
		hydra::Vector4R Jpsi = hydra::get<1>(event);
		hydra::Vector4R K    = hydra::get<2>(event);
		hydra::Vector4R pi   = hydra::get<3>(event);

		double M2_Jpsi_pi = (Jpsi + pi).mass2();
		double M2_Kpi     = (K + pi).mass2();

		return hydra::make_tuple( M2_Jpsi_pi, M2_Kpi);	
	}
);



Phase-space MC integrator
-------------------------

Other features
--------------