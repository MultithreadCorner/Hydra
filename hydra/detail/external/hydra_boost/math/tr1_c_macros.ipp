// Copyright John Maddock 2008-11.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HYDRA_BOOST_MATH_C_MACROS_IPP
#define HYDRA_BOOST_MATH_C_MACROS_IPP

// C99 Functions:
#ifdef acosh
#undef acosh
#endif
#define acosh hydra_boost_acosh
#ifdef acoshf
#undef acoshf
#endif
#define acoshf hydra_boost_acoshf
#ifdef acoshl
#undef acoshl
#endif
#define acoshl hydra_boost_acoshl

#ifdef asinh
#undef asinh
#endif
#define asinh hydra_boost_asinh
#ifdef asinhf
#undef asinhf
#endif
#define asinhf hydra_boost_asinhf
#ifdef asinhl
#undef asinhl
#endif
#define asinhl hydra_boost_asinhl

#ifdef atanh
#undef atanh
#endif
#define atanh hydra_boost_atanh
#ifdef atanhf
#undef atanhf
#endif
#define atanhf hydra_boost_atanhf
#ifdef atanhl
#undef atanhl
#endif
#define atanhl hydra_boost_atanhl

#ifdef cbrt
#undef cbrt
#endif
#define cbrt hydra_boost_cbrt
#ifdef cbrtf
#undef cbrtf
#endif
#define cbrtf hydra_boost_cbrtf
#ifdef cbrtl
#undef cbrtl
#endif
#define cbrtl hydra_boost_cbrtl

#ifdef copysign
#undef copysign
#endif
#define copysign hydra_boost_copysign
#ifdef copysignf
#undef copysignf
#endif
#define copysignf hydra_boost_copysignf
#ifdef copysignl
#undef copysignl
#endif
#define copysignl hydra_boost_copysignl

#ifdef erf
#undef erf
#endif
#define erf hydra_boost_erf
#ifdef erff
#undef erff
#endif
#define erff hydra_boost_erff
#ifdef erfl
#undef erfl
#endif
#define erfl hydra_boost_erfl

#ifdef erfc
#undef erfc
#endif
#define erfc hydra_boost_erfc
#ifdef erfcf
#undef erfcf
#endif
#define erfcf hydra_boost_erfcf
#ifdef erfcl
#undef erfcl
#endif
#define erfcl hydra_boost_erfcl

#if 0
#ifdef exp2
#undef exp2
#endif
#define exp2 hydra_boost_exp2
#ifdef exp2f
#undef exp2f
#endif
#define exp2f hydra_boost_exp2f
#ifdef exp2l
#undef exp2l
#endif
#define exp2l hydra_boost_exp2l
#endif

#ifdef expm1
#undef expm1
#endif
#define expm1 hydra_boost_expm1
#ifdef expm1f
#undef expm1f
#endif
#define expm1f hydra_boost_expm1f
#ifdef expm1l
#undef expm1l
#endif
#define expm1l hydra_boost_expm1l

#if 0
#ifdef fdim
#undef fdim
#endif
#define fdim hydra_boost_fdim
#ifdef fdimf
#undef fdimf
#endif
#define fdimf hydra_boost_fdimf
#ifdef fdiml
#undef fdiml
#endif
#define fdiml hydra_boost_fdiml
#ifdef acosh
#undef acosh
#endif
#define fma hydra_boost_fma
#ifdef fmaf
#undef fmaf
#endif
#define fmaf hydra_boost_fmaf
#ifdef fmal
#undef fmal
#endif
#define fmal hydra_boost_fmal
#endif

#ifdef fmax
#undef fmax
#endif
#define fmax hydra_boost_fmax
#ifdef fmaxf
#undef fmaxf
#endif
#define fmaxf hydra_boost_fmaxf
#ifdef fmaxl
#undef fmaxl
#endif
#define fmaxl hydra_boost_fmaxl

#ifdef fmin
#undef fmin
#endif
#define fmin hydra_boost_fmin
#ifdef fminf
#undef fminf
#endif
#define fminf hydra_boost_fminf
#ifdef fminl
#undef fminl
#endif
#define fminl hydra_boost_fminl

#ifdef hypot
#undef hypot
#endif
#define hypot hydra_boost_hypot
#ifdef hypotf
#undef hypotf
#endif
#define hypotf hydra_boost_hypotf
#ifdef hypotl
#undef hypotl
#endif
#define hypotl hydra_boost_hypotl

#if 0
#ifdef ilogb
#undef ilogb
#endif
#define ilogb hydra_boost_ilogb
#ifdef ilogbf
#undef ilogbf
#endif
#define ilogbf hydra_boost_ilogbf
#ifdef ilogbl
#undef ilogbl
#endif
#define ilogbl hydra_boost_ilogbl
#endif

#ifdef lgamma
#undef lgamma
#endif
#define lgamma hydra_boost_lgamma
#ifdef lgammaf
#undef lgammaf
#endif
#define lgammaf hydra_boost_lgammaf
#ifdef lgammal
#undef lgammal
#endif
#define lgammal hydra_boost_lgammal

#ifdef HYDRA_BOOST_HAS_LONG_LONG
#if 0
#ifdef llrint
#undef llrint
#endif
#define llrint hydra_boost_llrint
#ifdef llrintf
#undef llrintf
#endif
#define llrintf hydra_boost_llrintf
#ifdef llrintl
#undef llrintl
#endif
#define llrintl hydra_boost_llrintl
#endif
#ifdef llround
#undef llround
#endif
#define llround hydra_boost_llround
#ifdef llroundf
#undef llroundf
#endif
#define llroundf hydra_boost_llroundf
#ifdef llroundl
#undef llroundl
#endif
#define llroundl hydra_boost_llroundl
#endif

#ifdef log1p
#undef log1p
#endif
#define log1p hydra_boost_log1p
#ifdef log1pf
#undef log1pf
#endif
#define log1pf hydra_boost_log1pf
#ifdef log1pl
#undef log1pl
#endif
#define log1pl hydra_boost_log1pl

#if 0
#ifdef log2
#undef log2
#endif
#define log2 hydra_boost_log2
#ifdef log2f
#undef log2f
#endif
#define log2f hydra_boost_log2f
#ifdef log2l
#undef log2l
#endif
#define log2l hydra_boost_log2l

#ifdef logb
#undef logb
#endif
#define logb hydra_boost_logb
#ifdef logbf
#undef logbf
#endif
#define logbf hydra_boost_logbf
#ifdef logbl
#undef logbl
#endif
#define logbl hydra_boost_logbl

#ifdef lrint
#undef lrint
#endif
#define lrint hydra_boost_lrint
#ifdef lrintf
#undef lrintf
#endif
#define lrintf hydra_boost_lrintf
#ifdef lrintl
#undef lrintl
#endif
#define lrintl hydra_boost_lrintl
#endif

#ifdef lround
#undef lround
#endif
#define lround hydra_boost_lround
#ifdef lroundf
#undef lroundf
#endif
#define lroundf hydra_boost_lroundf
#ifdef lroundl
#undef lroundl
#endif
#define lroundl hydra_boost_lroundl

#if 0
#ifdef nan
#undef nan
#endif
#define nan hydra_boost_nan
#ifdef nanf
#undef nanf
#endif
#define nanf hydra_boost_nanf
#ifdef nanl
#undef nanl
#endif
#define nanl hydra_boost_nanl

#ifdef nearbyint
#undef nearbyint
#endif
#define nearbyint hydra_boost_nearbyint
#ifdef nearbyintf
#undef nearbyintf
#endif
#define nearbyintf hydra_boost_nearbyintf
#ifdef nearbyintl
#undef nearbyintl
#endif
#define nearbyintl hydra_boost_nearbyintl
#endif

#ifdef nextafter
#undef nextafter
#endif
#define nextafter hydra_boost_nextafter
#ifdef nextafterf
#undef nextafterf
#endif
#define nextafterf hydra_boost_nextafterf
#ifdef nextafterl
#undef nextafterl
#endif
#define nextafterl hydra_boost_nextafterl

#ifdef nexttoward
#undef nexttoward
#endif
#define nexttoward hydra_boost_nexttoward
#ifdef nexttowardf
#undef nexttowardf
#endif
#define nexttowardf hydra_boost_nexttowardf
#ifdef nexttowardl
#undef nexttowardl
#endif
#define nexttowardl hydra_boost_nexttowardl

#if 0
#ifdef remainder
#undef remainder
#endif
#define remainder hydra_boost_remainder
#ifdef remainderf
#undef remainderf
#endif
#define remainderf hydra_boost_remainderf
#ifdef remainderl
#undef remainderl
#endif
#define remainderl hydra_boost_remainderl

#ifdef remquo
#undef remquo
#endif
#define remquo hydra_boost_remquo
#ifdef remquof
#undef remquof
#endif
#define remquof hydra_boost_remquof
#ifdef remquol
#undef remquol
#endif
#define remquol hydra_boost_remquol

#ifdef rint
#undef rint
#endif
#define rint hydra_boost_rint
#ifdef rintf
#undef rintf
#endif
#define rintf hydra_boost_rintf
#ifdef rintl
#undef rintl
#endif
#define rintl hydra_boost_rintl
#endif

#ifdef round
#undef round
#endif
#define round hydra_boost_round
#ifdef roundf
#undef roundf
#endif
#define roundf hydra_boost_roundf
#ifdef roundl
#undef roundl
#endif
#define roundl hydra_boost_roundl

#if 0
#ifdef scalbln
#undef scalbln
#endif
#define scalbln hydra_boost_scalbln
#ifdef scalblnf
#undef scalblnf
#endif
#define scalblnf hydra_boost_scalblnf
#ifdef scalblnl
#undef scalblnl
#endif
#define scalblnl hydra_boost_scalblnl

#ifdef scalbn
#undef scalbn
#endif
#define scalbn hydra_boost_scalbn
#ifdef scalbnf
#undef scalbnf
#endif
#define scalbnf hydra_boost_scalbnf
#ifdef scalbnl
#undef scalbnl
#endif
#define scalbnl hydra_boost_scalbnl
#endif

#ifdef tgamma
#undef tgamma
#endif
#define tgamma hydra_boost_tgamma
#ifdef tgammaf
#undef tgammaf
#endif
#define tgammaf hydra_boost_tgammaf
#ifdef tgammal
#undef tgammal
#endif
#define tgammal hydra_boost_tgammal

#ifdef trunc
#undef trunc
#endif
#define trunc hydra_boost_trunc
#ifdef truncf
#undef truncf
#endif
#define truncf hydra_boost_truncf
#ifdef truncl
#undef truncl
#endif
#define truncl hydra_boost_truncl

// [5.2.1.1] associated Laguerre polynomials:
#ifdef assoc_laguerre
#undef assoc_laguerre
#endif
#define assoc_laguerre hydra_boost_assoc_laguerre
#ifdef assoc_laguerref
#undef assoc_laguerref
#endif
#define assoc_laguerref hydra_boost_assoc_laguerref
#ifdef assoc_laguerrel
#undef assoc_laguerrel
#endif
#define assoc_laguerrel hydra_boost_assoc_laguerrel

// [5.2.1.2] associated Legendre functions:
#ifdef assoc_legendre
#undef assoc_legendre
#endif
#define assoc_legendre hydra_boost_assoc_legendre
#ifdef assoc_legendref
#undef assoc_legendref
#endif
#define assoc_legendref hydra_boost_assoc_legendref
#ifdef assoc_legendrel
#undef assoc_legendrel
#endif
#define assoc_legendrel hydra_boost_assoc_legendrel

// [5.2.1.3] beta function:
#ifdef beta
#undef beta
#endif
#define beta hydra_boost_beta
#ifdef betaf
#undef betaf
#endif
#define betaf hydra_boost_betaf
#ifdef betal
#undef betal
#endif
#define betal hydra_boost_betal

// [5.2.1.4] (complete) elliptic integral of the first kind:
#ifdef comp_ellint_1
#undef comp_ellint_1
#endif
#define comp_ellint_1 hydra_boost_comp_ellint_1
#ifdef comp_ellint_1f
#undef comp_ellint_1f
#endif
#define comp_ellint_1f hydra_boost_comp_ellint_1f
#ifdef comp_ellint_1l
#undef comp_ellint_1l
#endif
#define comp_ellint_1l hydra_boost_comp_ellint_1l

// [5.2.1.5] (complete) elliptic integral of the second kind:
#ifdef comp_ellint_2
#undef comp_ellint_2
#endif
#define comp_ellint_2 hydra_boost_comp_ellint_2
#ifdef comp_ellint_2f
#undef comp_ellint_2f
#endif
#define comp_ellint_2f hydra_boost_comp_ellint_2f
#ifdef comp_ellint_2l
#undef comp_ellint_2l
#endif
#define comp_ellint_2l hydra_boost_comp_ellint_2l

// [5.2.1.6] (complete) elliptic integral of the third kind:
#ifdef comp_ellint_3
#undef comp_ellint_3
#endif
#define comp_ellint_3 hydra_boost_comp_ellint_3
#ifdef comp_ellint_3f
#undef comp_ellint_3f
#endif
#define comp_ellint_3f hydra_boost_comp_ellint_3f
#ifdef comp_ellint_3l
#undef comp_ellint_3l
#endif
#define comp_ellint_3l hydra_boost_comp_ellint_3l

#if 0
// [5.2.1.7] confluent hypergeometric functions:
#ifdef conf_hyper
#undef conf_hyper
#endif
#define conf_hyper hydra_boost_conf_hyper
#ifdef conf_hyperf
#undef conf_hyperf
#endif
#define conf_hyperf hydra_boost_conf_hyperf
#ifdef conf_hyperl
#undef conf_hyperl
#endif
#define conf_hyperl hydra_boost_conf_hyperl
#endif

// [5.2.1.8] regular modified cylindrical Bessel functions:
#ifdef cyl_bessel_i
#undef cyl_bessel_i
#endif
#define cyl_bessel_i hydra_boost_cyl_bessel_i
#ifdef cyl_bessel_if
#undef cyl_bessel_if
#endif
#define cyl_bessel_if hydra_boost_cyl_bessel_if
#ifdef cyl_bessel_il
#undef cyl_bessel_il
#endif
#define cyl_bessel_il hydra_boost_cyl_bessel_il

// [5.2.1.9] cylindrical Bessel functions (of the first kind):
#ifdef cyl_bessel_j
#undef cyl_bessel_j
#endif
#define cyl_bessel_j hydra_boost_cyl_bessel_j
#ifdef cyl_bessel_jf
#undef cyl_bessel_jf
#endif
#define cyl_bessel_jf hydra_boost_cyl_bessel_jf
#ifdef cyl_bessel_jl
#undef cyl_bessel_jl
#endif
#define cyl_bessel_jl hydra_boost_cyl_bessel_jl

// [5.2.1.10] irregular modified cylindrical Bessel functions:
#ifdef cyl_bessel_k
#undef cyl_bessel_k
#endif
#define cyl_bessel_k hydra_boost_cyl_bessel_k
#ifdef cyl_bessel_kf
#undef cyl_bessel_kf
#endif
#define cyl_bessel_kf hydra_boost_cyl_bessel_kf
#ifdef cyl_bessel_kl
#undef cyl_bessel_kl
#endif
#define cyl_bessel_kl hydra_boost_cyl_bessel_kl

// [5.2.1.11] cylindrical Neumann functions HYDRA_BOOST_MATH_C99_THROW_SPEC;
// cylindrical Bessel functions (of the second kind):
#ifdef cyl_neumann
#undef cyl_neumann
#endif
#define cyl_neumann hydra_boost_cyl_neumann
#ifdef cyl_neumannf
#undef cyl_neumannf
#endif
#define cyl_neumannf hydra_boost_cyl_neumannf
#ifdef cyl_neumannl
#undef cyl_neumannl
#endif
#define cyl_neumannl hydra_boost_cyl_neumannl

// [5.2.1.12] (incomplete) elliptic integral of the first kind:
#ifdef ellint_1
#undef ellint_1
#endif
#define ellint_1 hydra_boost_ellint_1
#ifdef ellint_1f
#undef ellint_1f
#endif
#define ellint_1f hydra_boost_ellint_1f
#ifdef ellint_1l
#undef ellint_1l
#endif
#define ellint_1l hydra_boost_ellint_1l

// [5.2.1.13] (incomplete) elliptic integral of the second kind:
#ifdef ellint_2
#undef ellint_2
#endif
#define ellint_2 hydra_boost_ellint_2
#ifdef ellint_2f
#undef ellint_2f
#endif
#define ellint_2f hydra_boost_ellint_2f
#ifdef ellint_2l
#undef ellint_2l
#endif
#define ellint_2l hydra_boost_ellint_2l

// [5.2.1.14] (incomplete) elliptic integral of the third kind:
#ifdef ellint_3
#undef ellint_3
#endif
#define ellint_3 hydra_boost_ellint_3
#ifdef ellint_3f
#undef ellint_3f
#endif
#define ellint_3f hydra_boost_ellint_3f
#ifdef ellint_3l
#undef ellint_3l
#endif
#define ellint_3l hydra_boost_ellint_3l

// [5.2.1.15] exponential integral:
#ifdef expint
#undef expint
#endif
#define expint hydra_boost_expint
#ifdef expintf
#undef expintf
#endif
#define expintf hydra_boost_expintf
#ifdef expintl
#undef expintl
#endif
#define expintl hydra_boost_expintl

// [5.2.1.16] Hermite polynomials:
#ifdef hermite
#undef hermite
#endif
#define hermite hydra_boost_hermite
#ifdef hermitef
#undef hermitef
#endif
#define hermitef hydra_boost_hermitef
#ifdef hermitel
#undef hermitel
#endif
#define hermitel hydra_boost_hermitel

#if 0
// [5.2.1.17] hypergeometric functions:
#ifdef hyperg
#undef hyperg
#endif
#define hyperg hydra_boost_hyperg
#ifdef hypergf
#undef hypergf
#endif
#define hypergf hydra_boost_hypergf
#ifdef hypergl
#undef hypergl
#endif
#define hypergl hydra_boost_hypergl
#endif

// [5.2.1.18] Laguerre polynomials:
#ifdef laguerre
#undef laguerre
#endif
#define laguerre hydra_boost_laguerre
#ifdef laguerref
#undef laguerref
#endif
#define laguerref hydra_boost_laguerref
#ifdef laguerrel
#undef laguerrel
#endif
#define laguerrel hydra_boost_laguerrel

// [5.2.1.19] Legendre polynomials:
#ifdef legendre
#undef legendre
#endif
#define legendre hydra_boost_legendre
#ifdef legendref
#undef legendref
#endif
#define legendref hydra_boost_legendref
#ifdef legendrel
#undef legendrel
#endif
#define legendrel hydra_boost_legendrel

// [5.2.1.20] Riemann zeta function:
#ifdef riemann_zeta
#undef riemann_zeta
#endif
#define riemann_zeta hydra_boost_riemann_zeta
#ifdef riemann_zetaf
#undef riemann_zetaf
#endif
#define riemann_zetaf hydra_boost_riemann_zetaf
#ifdef riemann_zetal
#undef riemann_zetal
#endif
#define riemann_zetal hydra_boost_riemann_zetal

// [5.2.1.21] spherical Bessel functions (of the first kind):
#ifdef sph_bessel
#undef sph_bessel
#endif
#define sph_bessel hydra_boost_sph_bessel
#ifdef sph_besself
#undef sph_besself
#endif
#define sph_besself hydra_boost_sph_besself
#ifdef sph_bessell
#undef sph_bessell
#endif
#define sph_bessell hydra_boost_sph_bessell

// [5.2.1.22] spherical associated Legendre functions:
#ifdef sph_legendre
#undef sph_legendre
#endif
#define sph_legendre hydra_boost_sph_legendre
#ifdef sph_legendref
#undef sph_legendref
#endif
#define sph_legendref hydra_boost_sph_legendref
#ifdef sph_legendrel
#undef sph_legendrel
#endif
#define sph_legendrel hydra_boost_sph_legendrel

// [5.2.1.23] spherical Neumann functions HYDRA_BOOST_MATH_C99_THROW_SPEC;
// spherical Bessel functions (of the second kind):
#ifdef sph_neumann
#undef sph_neumann
#endif
#define sph_neumann hydra_boost_sph_neumann
#ifdef sph_neumannf
#undef sph_neumannf
#endif
#define sph_neumannf hydra_boost_sph_neumannf
#ifdef sph_neumannl
#undef sph_neumannl
#endif
#define sph_neumannl hydra_boost_sph_neumannl

#endif // HYDRA_BOOST_MATH_C_MACROS_IPP
